//! Real CUDA GPU backend using dynamic library loading.
//!
//! Loads `libcuda.so` (Linux) or `nvcuda.dll` (Windows) at runtime via
//! `libloading` and calls the CUDA Driver API through resolved function
//! pointers.  No compile-time CUDA SDK dependency.

use std::ffi::CString;

use libloading::Library;
use outerlink_common::cuda_types::CuResult;
use outerlink_common::error::OuterLinkError;

use crate::gpu_backend::GpuBackend;

// ---------------------------------------------------------------------------
// VRAM safety
// ---------------------------------------------------------------------------

/// Minimum free VRAM (bytes) to keep as a safety margin.
/// Other applications may be using the GPU (desktop, ML inference, etc.).
pub const VRAM_SAFETY_MARGIN: usize = 512 * 1024 * 1024; // 512 MiB

/// Threshold (fraction of total) above which we log a warning.
const VRAM_WARN_THRESHOLD: f64 = 0.80;

// ---------------------------------------------------------------------------
// FFI type aliases for CUDA Driver API function pointers
// ---------------------------------------------------------------------------

// CUDA Driver API function signatures.  All return CUresult (i32).
// Opaque handles (CUcontext, CUmodule, etc.) are pointer-sized; we treat
// them as `*mut std::ffi::c_void` on the FFI boundary and cast to/from u64.

type FnCuInit = unsafe extern "C" fn(flags: u32) -> i32;
type FnCuDriverGetVersion = unsafe extern "C" fn(version: *mut i32) -> i32;

type FnCuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;
type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> i32;
type FnCuDeviceGetName = unsafe extern "C" fn(name: *mut u8, len: i32, dev: i32) -> i32;
type FnCuDeviceGetAttribute = unsafe extern "C" fn(pi: *mut i32, attrib: i32, dev: i32) -> i32;
type FnCuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, dev: i32) -> i32;

/// CUuuid is a struct { char bytes[16]; }
#[repr(C)]
struct CuUuidFfi {
    bytes: [u8; 16],
}
type FnCuDeviceGetUuid = unsafe extern "C" fn(uuid: *mut CuUuidFfi, dev: i32) -> i32;

// Context operations -- handles are opaque pointers.
type FnCuCtxCreate = unsafe extern "C" fn(pctx: *mut usize, flags: u32, dev: i32) -> i32;
type FnCuCtxDestroy = unsafe extern "C" fn(ctx: usize) -> i32;
type FnCuCtxSetCurrent = unsafe extern "C" fn(ctx: usize) -> i32;
type FnCuCtxPushCurrent = unsafe extern "C" fn(ctx: usize) -> i32;
type FnCuCtxPopCurrent = unsafe extern "C" fn(pctx: *mut usize) -> i32;
type FnCuCtxGetApiVersion = unsafe extern "C" fn(ctx: usize, version: *mut u32) -> i32;
type FnCuCtxGetLimit = unsafe extern "C" fn(pvalue: *mut u64, limit: i32) -> i32;
type FnCuCtxSetLimit = unsafe extern "C" fn(limit: i32, value: u64) -> i32;
type FnCuCtxGetStreamPriorityRange = unsafe extern "C" fn(least: *mut i32, greatest: *mut i32) -> i32;
type FnCuCtxGetFlags = unsafe extern "C" fn(flags: *mut u32) -> i32;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> i32;
type FnCuCtxGetDevice = unsafe extern "C" fn(device: *mut i32) -> i32;

// Memory operations
type FnCuMemAlloc = unsafe extern "C" fn(dptr: *mut u64, bytesize: usize) -> i32;
type FnCuMemFree = unsafe extern "C" fn(dptr: u64) -> i32;
type FnCuMemcpyHtoD = unsafe extern "C" fn(dst: u64, src: *const u8, bytecount: usize) -> i32;
type FnCuMemcpyDtoH = unsafe extern "C" fn(dst: *mut u8, src: u64, bytecount: usize) -> i32;
type FnCuMemGetInfo = unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> i32;

// Module operations
type FnCuModuleLoadData = unsafe extern "C" fn(module: *mut usize, image: *const u8) -> i32;
type FnCuModuleLoadDataEx = unsafe extern "C" fn(
    module: *mut usize,
    image: *const u8,
    num_options: u32,
    options: *const i32,
    option_values: *mut *mut std::ffi::c_void,
) -> i32;
type FnCuModuleUnload = unsafe extern "C" fn(module: usize) -> i32;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(hfunc: *mut usize, module: usize, name: *const u8) -> i32;
type FnCuModuleGetGlobal =
    unsafe extern "C" fn(dptr: *mut u64, size: *mut usize, module: usize, name: *const u8) -> i32;
type FnCuFuncGetAttribute = unsafe extern "C" fn(pi: *mut i32, attrib: i32, hfunc: usize) -> i32;
type FnCuFuncSetAttribute = unsafe extern "C" fn(hfunc: usize, attrib: i32, value: i32) -> i32;
type FnCuCtxGetCacheConfig = unsafe extern "C" fn(pconfig: *mut i32) -> i32;
type FnCuCtxSetCacheConfig = unsafe extern "C" fn(config: i32) -> i32;
type FnCuCtxGetSharedMemConfig = unsafe extern "C" fn(pconfig: *mut i32) -> i32;
type FnCuCtxSetSharedMemConfig = unsafe extern "C" fn(config: i32) -> i32;
type FnCuFuncSetCacheConfig = unsafe extern "C" fn(hfunc: usize, config: i32) -> i32;
type FnCuFuncSetSharedMemConfig = unsafe extern "C" fn(hfunc: usize, config: i32) -> i32;

// Memory address range query
type FnCuMemGetAddressRange = unsafe extern "C" fn(pbase: *mut u64, psize: *mut usize, dptr: u64) -> i32;

// Stream operations
type FnCuStreamCreate = unsafe extern "C" fn(stream: *mut usize, flags: u32) -> i32;
type FnCuStreamCreateWithPriority = unsafe extern "C" fn(stream: *mut usize, flags: u32, priority: i32) -> i32;
type FnCuStreamDestroy = unsafe extern "C" fn(stream: usize) -> i32;
type FnCuStreamSynchronize = unsafe extern "C" fn(stream: usize) -> i32;
type FnCuStreamQuery = unsafe extern "C" fn(stream: usize) -> i32;
type FnCuStreamGetPriority = unsafe extern "C" fn(stream: usize, priority: *mut i32) -> i32;
type FnCuStreamGetFlags = unsafe extern "C" fn(stream: usize, flags: *mut u32) -> i32;
type FnCuStreamGetCtx = unsafe extern "C" fn(stream: usize, pctx: *mut usize) -> i32;

// Stream capture / graph operations
type FnCuStreamBeginCapture = unsafe extern "C" fn(stream: usize, mode: i32) -> i32;
type FnCuStreamEndCapture = unsafe extern "C" fn(stream: usize, graph: *mut usize) -> i32;
type FnCuStreamIsCapturing = unsafe extern "C" fn(stream: usize, status: *mut i32) -> i32;
type FnCuStreamGetCaptureInfo = unsafe extern "C" fn(stream: usize, status: *mut i32, id: *mut u64, graph: *mut usize, deps: *mut *const usize, num_deps: *mut usize) -> i32;
type FnCuGraphCreate = unsafe extern "C" fn(graph: *mut usize, flags: u32) -> i32;
type FnCuGraphInstantiate = unsafe extern "C" fn(exec: *mut usize, graph: usize, err_node: *mut usize, log_buf: *mut u8, buf_size: usize) -> i32;
type FnCuGraphInstantiateWithFlags = unsafe extern "C" fn(exec: *mut usize, graph: usize, flags: u64) -> i32;
type FnCuGraphLaunch = unsafe extern "C" fn(exec: usize, stream: usize) -> i32;
type FnCuGraphExecDestroy = unsafe extern "C" fn(exec: usize) -> i32;
type FnCuGraphDestroy = unsafe extern "C" fn(graph: usize) -> i32;

// Event operations
type FnCuEventCreate = unsafe extern "C" fn(event: *mut usize, flags: u32) -> i32;
type FnCuEventDestroy = unsafe extern "C" fn(event: usize) -> i32;
type FnCuEventRecord = unsafe extern "C" fn(event: usize, stream: usize) -> i32;
type FnCuEventRecordWithFlags = unsafe extern "C" fn(event: usize, stream: usize, flags: u32) -> i32;
type FnCuEventSynchronize = unsafe extern "C" fn(event: usize) -> i32;
type FnCuEventElapsedTime =
    unsafe extern "C" fn(ms: *mut f32, start: usize, end: usize) -> i32;
type FnCuEventQuery = unsafe extern "C" fn(event: usize) -> i32;

// Async memory copy
type FnCuMemcpyHtoDAsync =
    unsafe extern "C" fn(dst: u64, src: *const u8, bytecount: usize, stream: usize) -> i32;
type FnCuMemcpyDtoHAsync =
    unsafe extern "C" fn(dst: *mut u8, src: u64, bytecount: usize, stream: usize) -> i32;

// Memset operations
type FnCuMemsetD8 = unsafe extern "C" fn(dst: u64, value: u8, count: usize) -> i32;
type FnCuMemsetD32 = unsafe extern "C" fn(dst: u64, value: u32, count: usize) -> i32;
type FnCuMemsetD8Async =
    unsafe extern "C" fn(dst: u64, value: u8, count: usize, stream: usize) -> i32;
type FnCuMemsetD32Async =
    unsafe extern "C" fn(dst: u64, value: u32, count: usize, stream: usize) -> i32;

// 16-bit memset
type FnCuMemsetD16 = unsafe extern "C" fn(dst: u64, value: u16, count: usize) -> i32;
type FnCuMemsetD16Async =
    unsafe extern "C" fn(dst: u64, value: u16, count: usize, stream: usize) -> i32;

// Device-to-device memory copy
type FnCuMemcpyDtoD = unsafe extern "C" fn(dst: u64, src: u64, bytecount: usize) -> i32;
type FnCuMemcpyDtoDAsync = unsafe extern "C" fn(dst: u64, src: u64, bytecount: usize, stream: usize) -> i32;

// Generic direction-agnostic memory copy
type FnCuMemcpy = unsafe extern "C" fn(dst: u64, src: u64, bytecount: usize) -> i32;
type FnCuMemcpyAsync =
    unsafe extern "C" fn(dst: u64, src: u64, bytecount: usize, stream: usize) -> i32;

// Host memory allocation with flags
type FnCuMemHostAlloc = unsafe extern "C" fn(pp: *mut *mut std::ffi::c_void, bytesize: usize, flags: u32) -> i32;

// 2D pitched allocation
type FnCuMemAllocPitch = unsafe extern "C" fn(dptr: *mut u64, pitch: *mut usize, width: usize, height: usize, element_size: u32) -> i32;

// Module loading
type FnCuModuleLoad = unsafe extern "C" fn(module: *mut usize, fname: *const u8) -> i32;
type FnCuModuleLoadFatBinary = unsafe extern "C" fn(module: *mut usize, fat_cubin: *const u8) -> i32;

// Managed / unified memory (CUDA 6.0+)
type FnCuMemAllocManaged = unsafe extern "C" fn(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
type FnCuMemPrefetchAsync = unsafe extern "C" fn(dptr: u64, count: usize, dst_device: i32, stream: usize) -> i32;
type FnCuMemAdvise = unsafe extern "C" fn(dptr: u64, count: usize, advice: i32, device: i32) -> i32;
type FnCuMemRangeGetAttribute = unsafe extern "C" fn(data: *mut std::ffi::c_void, data_size: usize, attribute: i32, dptr: u64, count: usize) -> i32;
type FnCuMemRangeGetAttributes = unsafe extern "C" fn(data: *mut *mut std::ffi::c_void, data_sizes: *const usize, attributes: *const i32, num_attributes: usize, dptr: u64, count: usize) -> i32;
// Library API (CUDA 12+) -- handles are opaque pointers (usize).
type FnCuLibraryLoadData = unsafe extern "C" fn(
    library: *mut usize,
    code: *const u8,
    jit_options: *const i32,
    jit_option_values: *mut *mut std::ffi::c_void,
    num_jit_options: u32,
    lib_options: *const i32,
    lib_option_values: *mut *mut std::ffi::c_void,
    num_lib_options: u32,
) -> i32;
type FnCuLibraryUnload = unsafe extern "C" fn(library: usize) -> i32;
type FnCuLibraryGetKernel = unsafe extern "C" fn(
    p_kernel: *mut usize,
    library: usize,
    name: *const u8,
) -> i32;
type FnCuLibraryGetModule = unsafe extern "C" fn(
    p_mod: *mut usize,
    library: usize,
) -> i32;
type FnCuKernelGetFunction = unsafe extern "C" fn(
    p_func: *mut usize,
    kernel: usize,
) -> i32;

// Stream-ordered memory / pool operations (CUDA 11.2+)
type FnCuMemAllocAsync = unsafe extern "C" fn(dptr: *mut u64, bytesize: usize, stream: usize) -> i32;
type FnCuMemFreeAsync = unsafe extern "C" fn(dptr: u64, stream: usize) -> i32;
type FnCuDeviceGetDefaultMemPool = unsafe extern "C" fn(pool: *mut usize, dev: i32) -> i32;
type FnCuDeviceGetMemPool = unsafe extern "C" fn(pool: *mut usize, dev: i32) -> i32;
type FnCuDeviceSetMemPool = unsafe extern "C" fn(dev: i32, pool: usize) -> i32;

/// CUmemAllocationGranularity_flags for cuMemGetAllocationGranularity.
type FnCuMemGetAllocationGranularity = unsafe extern "C" fn(granularity: *mut usize, prop: *const std::ffi::c_void, option: i32) -> i32;

/// CUmemPoolProps for cuMemPoolCreate (CUDA 11.2+).
#[repr(C)]
struct CuMemPoolProps {
    alloc_type: i32,         // CUmemAllocationType
    handle_types: i32,       // CUmemAllocationHandleType
    location_type: i32,      // CUmemLocationType
    location_id: i32,        // device ordinal
    win32_security: usize,   // pointer, NULL for us
    max_size: usize,         // 0 = no limit
    reserved: [u8; 56],      // reserved bytes per CUDA spec
}

type FnCuMemPoolCreate = unsafe extern "C" fn(pool: *mut usize, props: *const CuMemPoolProps) -> i32;
type FnCuMemPoolDestroy = unsafe extern "C" fn(pool: usize) -> i32;
type FnCuMemPoolGetAttribute = unsafe extern "C" fn(pool: usize, attr: i32, value: *mut u64) -> i32;
type FnCuMemPoolSetAttribute = unsafe extern "C" fn(pool: usize, attr: i32, value: *const u64) -> i32;
type FnCuMemPoolTrimTo = unsafe extern "C" fn(pool: usize, min_bytes_to_keep: usize) -> i32;
type FnCuMemAllocFromPoolAsync = unsafe extern "C" fn(dptr: *mut u64, bytesize: usize, pool: usize, stream: usize) -> i32;

// Host pinned memory
type FnCuMemAllocHost = unsafe extern "C" fn(pp: *mut *mut std::ffi::c_void, bytesize: usize) -> i32;
type FnCuMemFreeHost = unsafe extern "C" fn(p: *mut std::ffi::c_void) -> i32;
type FnCuMemHostGetDevicePointer = unsafe extern "C" fn(pdptr: *mut u64, p: *mut std::ffi::c_void, flags: u32) -> i32;
type FnCuMemHostGetFlags = unsafe extern "C" fn(p_flags: *mut u32, p: *mut std::ffi::c_void) -> i32;
type FnCuMemHostRegister = unsafe extern "C" fn(p: *mut std::ffi::c_void, bytesize: usize, flags: u32) -> i32;
type FnCuMemHostUnregister = unsafe extern "C" fn(p: *mut std::ffi::c_void) -> i32;

// Peer access
type FnCuDeviceCanAccessPeer = unsafe extern "C" fn(can_access: *mut i32, dev: i32, peer_dev: i32) -> i32;
type FnCuDeviceGetP2PAttribute = unsafe extern "C" fn(value: *mut i32, attrib: i32, src: i32, dst: i32) -> i32;
type FnCuCtxEnablePeerAccess = unsafe extern "C" fn(peer_ctx: usize, flags: u32) -> i32;
type FnCuCtxDisablePeerAccess = unsafe extern "C" fn(peer_ctx: usize) -> i32;

// Stream wait event
type FnCuStreamWaitEvent = unsafe extern "C" fn(stream: usize, event: usize, flags: u32) -> i32;

// Occupancy operations
type FnCuOccupancyMaxActiveBlocksPerMultiprocessor =
    unsafe extern "C" fn(num_blocks: *mut i32, func: usize, block_size: i32, dynamic_smem_size: usize) -> i32;
type FnCuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags =
    unsafe extern "C" fn(num_blocks: *mut i32, func: usize, block_size: i32, dynamic_smem_size: usize, flags: u32) -> i32;
type FnCuOccupancyMaxPotentialBlockSize =
    unsafe extern "C" fn(
        min_grid_size: *mut i32,
        block_size: *mut i32,
        func: usize,
        block_size_to_dynamic_smem_size: usize, // function pointer, NULL when called from server
        dynamic_smem_size: usize,
        block_size_limit: i32,
    ) -> i32;
type FnCuOccupancyMaxPotentialBlockSizeWithFlags =
    unsafe extern "C" fn(
        min_grid_size: *mut i32,
        block_size: *mut i32,
        func: usize,
        block_size_to_dynamic_smem_size: usize, // function pointer, NULL when called from server
        dynamic_smem_size: usize,
        block_size_limit: i32,
        flags: u32,
    ) -> i32;

// Pointer attribute queries
type FnCuPointerGetAttribute =
    unsafe extern "C" fn(data: *mut std::ffi::c_void, attribute: i32, ptr: u64) -> i32;
type FnCuPointerGetAttributes =
    unsafe extern "C" fn(num_attributes: u32, attributes: *const i32, data: *mut *mut std::ffi::c_void, ptr: u64) -> i32;

// Primary context operations
type FnCuDevicePrimaryCtxRetain = unsafe extern "C" fn(pctx: *mut usize, dev: i32) -> i32;
type FnCuDevicePrimaryCtxRelease = unsafe extern "C" fn(dev: i32) -> i32;
type FnCuDevicePrimaryCtxGetState = unsafe extern "C" fn(dev: i32, flags: *mut u32, active: *mut i32) -> i32;
type FnCuDevicePrimaryCtxSetFlags = unsafe extern "C" fn(dev: i32, flags: u32) -> i32;
type FnCuDevicePrimaryCtxReset = unsafe extern "C" fn(dev: i32) -> i32;

type FnCuDeviceGetPCIBusId = unsafe extern "C" fn(pci_bus_id: *mut u8, len: i32, dev: i32) -> i32;
type FnCuDeviceGetByPCIBusId = unsafe extern "C" fn(dev: *mut i32, pci_bus_id: *const u8) -> i32;

// Kernel launch
type FnCuLaunchKernel = unsafe extern "C" fn(
    f: usize,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: usize,
    kernel_params: *mut *mut std::ffi::c_void,
    extra: *mut *mut std::ffi::c_void,
) -> i32;

// cuLaunchCooperativeKernel has identical signature minus the `extra` parameter,
// but uses kernelParams only. We use the same type since the FFI signature matches.
type FnCuLaunchCooperativeKernel = unsafe extern "C" fn(
    f: usize,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: usize,
    kernel_params: *mut *mut std::ffi::c_void,
) -> i32;

// ---------------------------------------------------------------------------
// CudaApi: resolved function pointer table
// ---------------------------------------------------------------------------

/// Holds resolved function pointers from the CUDA driver library.
///
/// Each field is `Option<...>` so we can gracefully handle missing symbols
/// (e.g. older driver versions that lack certain _v2 functions).
struct CudaApi {
    cu_init: Option<FnCuInit>,
    cu_driver_get_version: Option<FnCuDriverGetVersion>,

    cu_device_get: Option<FnCuDeviceGet>,
    cu_device_get_count: Option<FnCuDeviceGetCount>,
    cu_device_get_name: Option<FnCuDeviceGetName>,
    cu_device_get_attribute: Option<FnCuDeviceGetAttribute>,
    cu_device_total_mem: Option<FnCuDeviceTotalMem>,
    cu_device_get_uuid: Option<FnCuDeviceGetUuid>,

    cu_ctx_create: Option<FnCuCtxCreate>,
    cu_ctx_destroy: Option<FnCuCtxDestroy>,
    cu_ctx_set_current: Option<FnCuCtxSetCurrent>,
    cu_ctx_push_current: Option<FnCuCtxPushCurrent>,
    cu_ctx_pop_current: Option<FnCuCtxPopCurrent>,
    cu_ctx_get_api_version: Option<FnCuCtxGetApiVersion>,
    cu_ctx_get_limit: Option<FnCuCtxGetLimit>,
    cu_ctx_set_limit: Option<FnCuCtxSetLimit>,
    cu_ctx_get_stream_priority_range: Option<FnCuCtxGetStreamPriorityRange>,
    cu_ctx_get_flags: Option<FnCuCtxGetFlags>,
    cu_ctx_synchronize: Option<FnCuCtxSynchronize>,
    cu_ctx_get_device: Option<FnCuCtxGetDevice>,

    cu_mem_alloc: Option<FnCuMemAlloc>,
    cu_mem_free: Option<FnCuMemFree>,
    cu_memcpy_htod: Option<FnCuMemcpyHtoD>,
    cu_memcpy_dtoh: Option<FnCuMemcpyDtoH>,
    cu_mem_get_info: Option<FnCuMemGetInfo>,
    cu_memcpy_dtod: Option<FnCuMemcpyDtoD>,
    cu_memcpy_dtod_async: Option<FnCuMemcpyDtoDAsync>,
    cu_memcpy: Option<FnCuMemcpy>,
    cu_memcpy_async: Option<FnCuMemcpyAsync>,
    cu_memcpy_htod_async: Option<FnCuMemcpyHtoDAsync>,
    cu_memcpy_dtoh_async: Option<FnCuMemcpyDtoHAsync>,
    cu_memset_d8: Option<FnCuMemsetD8>,
    cu_memset_d32: Option<FnCuMemsetD32>,
    cu_memset_d8_async: Option<FnCuMemsetD8Async>,
    cu_memset_d32_async: Option<FnCuMemsetD32Async>,
    cu_memset_d16: Option<FnCuMemsetD16>,
    cu_memset_d16_async: Option<FnCuMemsetD16Async>,
    // Managed / unified memory
    cu_mem_alloc_managed: Option<FnCuMemAllocManaged>,
    cu_mem_prefetch_async: Option<FnCuMemPrefetchAsync>,
    cu_mem_advise: Option<FnCuMemAdvise>,
    cu_mem_range_get_attribute: Option<FnCuMemRangeGetAttribute>,
    cu_mem_range_get_attributes: Option<FnCuMemRangeGetAttributes>,
    // Stream-ordered memory / pool (CUDA 11.2+)
    cu_mem_alloc_async: Option<FnCuMemAllocAsync>,
    cu_mem_free_async: Option<FnCuMemFreeAsync>,
    cu_device_get_default_mem_pool: Option<FnCuDeviceGetDefaultMemPool>,
    cu_mem_pool_create: Option<FnCuMemPoolCreate>,
    cu_mem_pool_destroy: Option<FnCuMemPoolDestroy>,
    cu_mem_pool_get_attribute: Option<FnCuMemPoolGetAttribute>,
    cu_mem_pool_set_attribute: Option<FnCuMemPoolSetAttribute>,
    cu_mem_pool_trim_to: Option<FnCuMemPoolTrimTo>,
    cu_mem_alloc_from_pool_async: Option<FnCuMemAllocFromPoolAsync>,
    cu_device_get_mem_pool: Option<FnCuDeviceGetMemPool>,
    cu_device_set_mem_pool: Option<FnCuDeviceSetMemPool>,
    cu_mem_get_allocation_granularity: Option<FnCuMemGetAllocationGranularity>,

    cu_mem_alloc_host: Option<FnCuMemAllocHost>,
    cu_mem_host_alloc: Option<FnCuMemHostAlloc>,
    cu_mem_free_host: Option<FnCuMemFreeHost>,
    cu_mem_host_get_device_pointer: Option<FnCuMemHostGetDevicePointer>,
    cu_mem_host_get_flags: Option<FnCuMemHostGetFlags>,
    cu_mem_host_register: Option<FnCuMemHostRegister>,
    cu_mem_host_unregister: Option<FnCuMemHostUnregister>,

    cu_mem_alloc_pitch: Option<FnCuMemAllocPitch>,

    cu_module_load: Option<FnCuModuleLoad>,
    cu_module_load_fat_binary: Option<FnCuModuleLoadFatBinary>,
    cu_module_load_data: Option<FnCuModuleLoadData>,
    cu_module_load_data_ex: Option<FnCuModuleLoadDataEx>,
    cu_module_unload: Option<FnCuModuleUnload>,
    cu_module_get_function: Option<FnCuModuleGetFunction>,
    cu_module_get_global: Option<FnCuModuleGetGlobal>,
    cu_func_get_attribute: Option<FnCuFuncGetAttribute>,
    cu_func_set_attribute: Option<FnCuFuncSetAttribute>,
    cu_ctx_get_cache_config: Option<FnCuCtxGetCacheConfig>,
    cu_ctx_set_cache_config: Option<FnCuCtxSetCacheConfig>,
    cu_ctx_get_shared_mem_config: Option<FnCuCtxGetSharedMemConfig>,
    cu_ctx_set_shared_mem_config: Option<FnCuCtxSetSharedMemConfig>,
    cu_func_set_cache_config: Option<FnCuFuncSetCacheConfig>,
    cu_func_set_shared_mem_config: Option<FnCuFuncSetSharedMemConfig>,
    cu_mem_get_address_range: Option<FnCuMemGetAddressRange>,

    cu_stream_create: Option<FnCuStreamCreate>,
    cu_stream_create_with_priority: Option<FnCuStreamCreateWithPriority>,
    cu_stream_destroy: Option<FnCuStreamDestroy>,
    cu_stream_synchronize: Option<FnCuStreamSynchronize>,
    cu_stream_query: Option<FnCuStreamQuery>,
    cu_stream_get_priority: Option<FnCuStreamGetPriority>,
    cu_stream_get_flags: Option<FnCuStreamGetFlags>,
    cu_stream_get_ctx: Option<FnCuStreamGetCtx>,
    cu_stream_wait_event: Option<FnCuStreamWaitEvent>,

    // Stream capture / graph operations
    cu_stream_begin_capture: Option<FnCuStreamBeginCapture>,
    cu_stream_end_capture: Option<FnCuStreamEndCapture>,
    cu_stream_is_capturing: Option<FnCuStreamIsCapturing>,
    cu_stream_get_capture_info: Option<FnCuStreamGetCaptureInfo>,
    cu_graph_create: Option<FnCuGraphCreate>,
    cu_graph_instantiate: Option<FnCuGraphInstantiate>,
    cu_graph_instantiate_with_flags: Option<FnCuGraphInstantiateWithFlags>,
    cu_graph_launch: Option<FnCuGraphLaunch>,
    cu_graph_exec_destroy: Option<FnCuGraphExecDestroy>,
    cu_graph_destroy: Option<FnCuGraphDestroy>,

    cu_event_create: Option<FnCuEventCreate>,
    cu_event_destroy: Option<FnCuEventDestroy>,
    cu_event_record: Option<FnCuEventRecord>,
    cu_event_record_with_flags: Option<FnCuEventRecordWithFlags>,
    cu_event_synchronize: Option<FnCuEventSynchronize>,
    cu_event_elapsed_time: Option<FnCuEventElapsedTime>,
    cu_event_query: Option<FnCuEventQuery>,

    cu_launch_kernel: Option<FnCuLaunchKernel>,
    cu_launch_cooperative_kernel: Option<FnCuLaunchCooperativeKernel>,
    cu_device_get_pci_bus_id: Option<FnCuDeviceGetPCIBusId>,
    cu_device_get_by_pci_bus_id: Option<FnCuDeviceGetByPCIBusId>,

    cu_occupancy_max_active_blocks: Option<FnCuOccupancyMaxActiveBlocksPerMultiprocessor>,
    cu_occupancy_max_active_blocks_with_flags: Option<FnCuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags>,
    cu_occupancy_max_potential_block_size: Option<FnCuOccupancyMaxPotentialBlockSize>,
    cu_occupancy_max_potential_block_size_with_flags: Option<FnCuOccupancyMaxPotentialBlockSizeWithFlags>,

    cu_pointer_get_attribute: Option<FnCuPointerGetAttribute>,
    cu_pointer_get_attributes: Option<FnCuPointerGetAttributes>,

    cu_device_can_access_peer: Option<FnCuDeviceCanAccessPeer>,
    cu_device_get_p2p_attribute: Option<FnCuDeviceGetP2PAttribute>,
    cu_ctx_enable_peer_access: Option<FnCuCtxEnablePeerAccess>,
    cu_ctx_disable_peer_access: Option<FnCuCtxDisablePeerAccess>,

    cu_device_primary_ctx_retain: Option<FnCuDevicePrimaryCtxRetain>,
    cu_device_primary_ctx_release: Option<FnCuDevicePrimaryCtxRelease>,
    cu_device_primary_ctx_get_state: Option<FnCuDevicePrimaryCtxGetState>,
    cu_device_primary_ctx_set_flags: Option<FnCuDevicePrimaryCtxSetFlags>,
    cu_device_primary_ctx_reset: Option<FnCuDevicePrimaryCtxReset>,

    // Library API (CUDA 12+)
    cu_library_load_data: Option<FnCuLibraryLoadData>,
    cu_library_unload: Option<FnCuLibraryUnload>,
    cu_library_get_kernel: Option<FnCuLibraryGetKernel>,
    cu_library_get_module: Option<FnCuLibraryGetModule>,
    cu_kernel_get_function: Option<FnCuKernelGetFunction>,
}

// Safety: The function pointers in CudaApi are plain function pointers
// (not closures) obtained from the loaded library. They are inherently
// Send + Sync as they point to immutable code in the shared library.
unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

/// Try to load a symbol from the library, returning `None` if not found.
///
/// Tries the primary name first, then the fallback (if provided).
/// This handles the _v2 versioning pattern in the CUDA Driver API.
macro_rules! load_sym {
    ($lib:expr, $primary:expr) => {{
        let sym = unsafe { $lib.get::<*const ()>($primary) };
        match sym {
            Ok(s) => Some(unsafe { std::mem::transmute(*s) }),
            Err(_) => {
                tracing::debug!(
                    "CUDA symbol {:?} not found, feature will be unavailable",
                    String::from_utf8_lossy($primary)
                );
                None
            }
        }
    }};
    ($lib:expr, $primary:expr, $fallback:expr) => {{
        let sym = unsafe { $lib.get::<*const ()>($primary) };
        match sym {
            Ok(s) => Some(unsafe { std::mem::transmute(*s) }),
            Err(_) => {
                // Try the fallback name.
                let sym2 = unsafe { $lib.get::<*const ()>($fallback) };
                match sym2 {
                    Ok(s2) => Some(unsafe { std::mem::transmute(*s2) }),
                    Err(_) => {
                        tracing::debug!(
                            "CUDA symbols {:?} / {:?} not found",
                            String::from_utf8_lossy($primary),
                            String::from_utf8_lossy($fallback),
                        );
                        None
                    }
                }
            }
        }
    }};
}

impl CudaApi {
    /// Resolve all known CUDA Driver API symbols from the loaded library.
    ///
    /// Missing symbols are logged and stored as `None` rather than causing
    /// an error, so the backend can still service operations whose symbols
    /// were found.
    fn load(lib: &Library) -> Self {
        Self {
            cu_init: load_sym!(lib, b"cuInit\0"),
            cu_driver_get_version: load_sym!(lib, b"cuDriverGetVersion\0"),

            cu_device_get: load_sym!(lib, b"cuDeviceGet\0"),
            cu_device_get_count: load_sym!(lib, b"cuDeviceGetCount\0"),
            cu_device_get_name: load_sym!(lib, b"cuDeviceGetName\0"),
            cu_device_get_attribute: load_sym!(lib, b"cuDeviceGetAttribute\0"),
            cu_device_total_mem: load_sym!(lib, b"cuDeviceTotalMem_v2\0", b"cuDeviceTotalMem\0"),
            cu_device_get_uuid: load_sym!(lib, b"cuDeviceGetUuid\0"),

            cu_ctx_create: load_sym!(lib, b"cuCtxCreate_v2\0", b"cuCtxCreate\0"),
            cu_ctx_destroy: load_sym!(lib, b"cuCtxDestroy_v2\0", b"cuCtxDestroy\0"),
            cu_ctx_set_current: load_sym!(lib, b"cuCtxSetCurrent\0"),
            cu_ctx_push_current: load_sym!(lib, b"cuCtxPushCurrent_v2\0", b"cuCtxPushCurrent\0"),
            cu_ctx_pop_current: load_sym!(lib, b"cuCtxPopCurrent_v2\0", b"cuCtxPopCurrent\0"),
            cu_ctx_get_api_version: load_sym!(lib, b"cuCtxGetApiVersion\0"),
            cu_ctx_get_limit: load_sym!(lib, b"cuCtxGetLimit\0"),
            cu_ctx_set_limit: load_sym!(lib, b"cuCtxSetLimit\0"),
            cu_ctx_get_stream_priority_range: load_sym!(lib, b"cuCtxGetStreamPriorityRange\0"),
            cu_ctx_get_flags: load_sym!(lib, b"cuCtxGetFlags\0"),
            cu_ctx_synchronize: load_sym!(lib, b"cuCtxSynchronize\0"),
            cu_ctx_get_device: load_sym!(lib, b"cuCtxGetDevice\0"),

            cu_mem_alloc: load_sym!(lib, b"cuMemAlloc_v2\0", b"cuMemAlloc\0"),
            cu_mem_free: load_sym!(lib, b"cuMemFree_v2\0", b"cuMemFree\0"),
            cu_memcpy_htod: load_sym!(lib, b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"),
            cu_memcpy_dtoh: load_sym!(lib, b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"),
            cu_mem_get_info: load_sym!(lib, b"cuMemGetInfo_v2\0", b"cuMemGetInfo\0"),
            cu_memcpy_dtod: load_sym!(lib, b"cuMemcpyDtoD_v2\0", b"cuMemcpyDtoD\0"),
            cu_memcpy_dtod_async: load_sym!(lib, b"cuMemcpyDtoDAsync_v2\0", b"cuMemcpyDtoDAsync\0"),
            cu_memcpy: load_sym!(lib, b"cuMemcpy\0"),
            cu_memcpy_async: load_sym!(lib, b"cuMemcpyAsync\0"),
            cu_memcpy_htod_async: load_sym!(lib, b"cuMemcpyHtoDAsync_v2\0", b"cuMemcpyHtoDAsync\0"),
            cu_memcpy_dtoh_async: load_sym!(lib, b"cuMemcpyDtoHAsync_v2\0", b"cuMemcpyDtoHAsync\0"),
            cu_memset_d8: load_sym!(lib, b"cuMemsetD8_v2\0", b"cuMemsetD8\0"),
            cu_memset_d32: load_sym!(lib, b"cuMemsetD32_v2\0", b"cuMemsetD32\0"),
            cu_memset_d8_async: load_sym!(lib, b"cuMemsetD8Async\0"),
            cu_memset_d32_async: load_sym!(lib, b"cuMemsetD32Async\0"),
            cu_memset_d16: load_sym!(lib, b"cuMemsetD16_v2\0", b"cuMemsetD16\0"),
            cu_memset_d16_async: load_sym!(lib, b"cuMemsetD16Async\0"),
            // Managed / unified memory (CUDA 6.0+)
            cu_mem_alloc_managed: load_sym!(lib, b"cuMemAllocManaged\0"),
            cu_mem_prefetch_async: load_sym!(lib, b"cuMemPrefetchAsync\0"),
            cu_mem_advise: load_sym!(lib, b"cuMemAdvise\0"),
            cu_mem_range_get_attribute: load_sym!(lib, b"cuMemRangeGetAttribute\0"),
            cu_mem_range_get_attributes: load_sym!(lib, b"cuMemRangeGetAttributes\0"),
            // Stream-ordered memory / pool (CUDA 11.2+, optional)
            cu_mem_alloc_async: load_sym!(lib, b"cuMemAllocAsync\0"),
            cu_mem_free_async: load_sym!(lib, b"cuMemFreeAsync\0"),
            cu_device_get_default_mem_pool: load_sym!(lib, b"cuDeviceGetDefaultMemPool\0"),
            cu_mem_pool_create: load_sym!(lib, b"cuMemPoolCreate\0"),
            cu_mem_pool_destroy: load_sym!(lib, b"cuMemPoolDestroy\0"),
            cu_mem_pool_get_attribute: load_sym!(lib, b"cuMemPoolGetAttribute\0"),
            cu_mem_pool_set_attribute: load_sym!(lib, b"cuMemPoolSetAttribute\0"),
            cu_mem_pool_trim_to: load_sym!(lib, b"cuMemPoolTrimTo\0"),
            cu_mem_alloc_from_pool_async: load_sym!(lib, b"cuMemAllocFromPoolAsync\0"),
            cu_device_get_mem_pool: load_sym!(lib, b"cuDeviceGetMemPool\0"),
            cu_device_set_mem_pool: load_sym!(lib, b"cuDeviceSetMemPool\0"),
            cu_mem_get_allocation_granularity: load_sym!(lib, b"cuMemGetAllocationGranularity\0"),

            cu_mem_alloc_host: load_sym!(lib, b"cuMemAllocHost_v2\0", b"cuMemAllocHost\0"),
            cu_mem_host_alloc: load_sym!(lib, b"cuMemHostAlloc\0"),
            cu_mem_free_host: load_sym!(lib, b"cuMemFreeHost\0"),
            cu_mem_host_get_device_pointer: load_sym!(lib, b"cuMemHostGetDevicePointer_v2\0", b"cuMemHostGetDevicePointer\0"),
            cu_mem_host_get_flags: load_sym!(lib, b"cuMemHostGetFlags\0"),
            cu_mem_host_register: load_sym!(lib, b"cuMemHostRegister_v2\0", b"cuMemHostRegister\0"),
            cu_mem_host_unregister: load_sym!(lib, b"cuMemHostUnregister\0"),

            cu_mem_alloc_pitch: load_sym!(lib, b"cuMemAllocPitch_v2\0", b"cuMemAllocPitch\0"),

            cu_module_load: load_sym!(lib, b"cuModuleLoad\0"),
            cu_module_load_fat_binary: load_sym!(lib, b"cuModuleLoadFatBinary\0"),
            cu_module_load_data: load_sym!(lib, b"cuModuleLoadData\0"),
            cu_module_load_data_ex: load_sym!(lib, b"cuModuleLoadDataEx\0"),
            cu_module_unload: load_sym!(lib, b"cuModuleUnload\0"),
            cu_module_get_function: load_sym!(lib, b"cuModuleGetFunction\0"),
            cu_module_get_global: load_sym!(lib, b"cuModuleGetGlobal_v2\0", b"cuModuleGetGlobal\0"),
            cu_func_get_attribute: load_sym!(lib, b"cuFuncGetAttribute\0"),
            cu_func_set_attribute: load_sym!(lib, b"cuFuncSetAttribute\0"),
            cu_ctx_get_cache_config: load_sym!(lib, b"cuCtxGetCacheConfig\0"),
            cu_ctx_set_cache_config: load_sym!(lib, b"cuCtxSetCacheConfig\0"),
            cu_ctx_get_shared_mem_config: load_sym!(lib, b"cuCtxGetSharedMemConfig\0"),
            cu_ctx_set_shared_mem_config: load_sym!(lib, b"cuCtxSetSharedMemConfig\0"),
            cu_func_set_cache_config: load_sym!(lib, b"cuFuncSetCacheConfig\0"),
            cu_func_set_shared_mem_config: load_sym!(lib, b"cuFuncSetSharedMemConfig\0"),
            cu_mem_get_address_range: load_sym!(lib, b"cuMemGetAddressRange_v2\0", b"cuMemGetAddressRange\0"),

            cu_stream_create: load_sym!(lib, b"cuStreamCreate\0"),
            cu_stream_create_with_priority: load_sym!(lib, b"cuStreamCreateWithPriority\0"),
            cu_stream_destroy: load_sym!(lib, b"cuStreamDestroy_v2\0", b"cuStreamDestroy\0"),
            cu_stream_synchronize: load_sym!(lib, b"cuStreamSynchronize\0"),
            cu_stream_query: load_sym!(lib, b"cuStreamQuery\0"),
            cu_stream_get_priority: load_sym!(lib, b"cuStreamGetPriority\0"),
            cu_stream_get_flags: load_sym!(lib, b"cuStreamGetFlags_v2\0", b"cuStreamGetFlags\0"),
            cu_stream_get_ctx: load_sym!(lib, b"cuStreamGetCtx_v2\0", b"cuStreamGetCtx\0"),
            cu_stream_wait_event: load_sym!(lib, b"cuStreamWaitEvent\0"),

            cu_stream_begin_capture: load_sym!(lib, b"cuStreamBeginCapture_v2\0", b"cuStreamBeginCapture\0"),
            cu_stream_end_capture: load_sym!(lib, b"cuStreamEndCapture\0"),
            cu_stream_is_capturing: load_sym!(lib, b"cuStreamIsCapturing\0"),
            cu_stream_get_capture_info: load_sym!(lib, b"cuStreamGetCaptureInfo_v2\0", b"cuStreamGetCaptureInfo\0"),
            cu_graph_create: load_sym!(lib, b"cuGraphCreate\0"),
            cu_graph_instantiate: load_sym!(lib, b"cuGraphInstantiate_v2\0", b"cuGraphInstantiate\0"),
            cu_graph_instantiate_with_flags: load_sym!(lib, b"cuGraphInstantiateWithFlags\0"),
            cu_graph_launch: load_sym!(lib, b"cuGraphLaunch\0"),
            cu_graph_exec_destroy: load_sym!(lib, b"cuGraphExecDestroy\0"),
            cu_graph_destroy: load_sym!(lib, b"cuGraphDestroy\0"),

            cu_event_create: load_sym!(lib, b"cuEventCreate\0"),
            cu_event_destroy: load_sym!(lib, b"cuEventDestroy_v2\0", b"cuEventDestroy\0"),
            cu_event_record: load_sym!(lib, b"cuEventRecord\0"),
            cu_event_record_with_flags: load_sym!(lib, b"cuEventRecordWithFlags\0"),
            cu_event_synchronize: load_sym!(lib, b"cuEventSynchronize\0"),
            cu_event_elapsed_time: load_sym!(lib, b"cuEventElapsedTime\0"),
            cu_event_query: load_sym!(lib, b"cuEventQuery\0"),

            cu_launch_kernel: load_sym!(lib, b"cuLaunchKernel\0"),
            cu_launch_cooperative_kernel: load_sym!(lib, b"cuLaunchCooperativeKernel\0"),
            cu_device_get_pci_bus_id: load_sym!(lib, b"cuDeviceGetPCIBusId\0"),
            cu_device_get_by_pci_bus_id: load_sym!(lib, b"cuDeviceGetByPCIBusId\0"),

            cu_occupancy_max_active_blocks: load_sym!(lib, b"cuOccupancyMaxActiveBlocksPerMultiprocessor\0"),
            cu_occupancy_max_active_blocks_with_flags: load_sym!(lib, b"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\0"),
            cu_occupancy_max_potential_block_size: load_sym!(lib, b"cuOccupancyMaxPotentialBlockSize\0"),
            cu_occupancy_max_potential_block_size_with_flags: load_sym!(lib, b"cuOccupancyMaxPotentialBlockSizeWithFlags\0"),

            cu_pointer_get_attribute: load_sym!(lib, b"cuPointerGetAttribute\0"),
            cu_pointer_get_attributes: load_sym!(lib, b"cuPointerGetAttributes\0"),

            cu_device_can_access_peer: load_sym!(lib, b"cuDeviceCanAccessPeer\0"),
            cu_device_get_p2p_attribute: load_sym!(lib, b"cuDeviceGetP2PAttribute\0"),
            cu_ctx_enable_peer_access: load_sym!(lib, b"cuCtxEnablePeerAccess\0"),
            cu_ctx_disable_peer_access: load_sym!(lib, b"cuCtxDisablePeerAccess\0"),

            cu_device_primary_ctx_retain: load_sym!(lib, b"cuDevicePrimaryCtxRetain\0"),
            cu_device_primary_ctx_release: load_sym!(lib, b"cuDevicePrimaryCtxRelease_v2\0", b"cuDevicePrimaryCtxRelease\0"),
            cu_device_primary_ctx_get_state: load_sym!(lib, b"cuDevicePrimaryCtxGetState\0"),
            cu_device_primary_ctx_set_flags: load_sym!(lib, b"cuDevicePrimaryCtxSetFlags_v2\0", b"cuDevicePrimaryCtxSetFlags\0"),
            cu_device_primary_ctx_reset: load_sym!(lib, b"cuDevicePrimaryCtxReset_v2\0", b"cuDevicePrimaryCtxReset\0"),

            // Library API (CUDA 12+) -- optional, not present on older drivers
            cu_library_load_data: load_sym!(lib, b"cuLibraryLoadData\0"),
            cu_library_unload: load_sym!(lib, b"cuLibraryUnload\0"),
            cu_library_get_kernel: load_sym!(lib, b"cuLibraryGetKernel\0"),
            cu_library_get_module: load_sym!(lib, b"cuLibraryGetModule\0"),
            cu_kernel_get_function: load_sym!(lib, b"cuKernelGetFunction\0"),
        }
    }
}

// ---------------------------------------------------------------------------
// Result mapping
// ---------------------------------------------------------------------------

/// Map a raw CUDA CUresult (i32) to our CuResult.
///
/// `CUDA_SUCCESS` (0) returns `Ok(())`, anything else returns `Err`.
fn map_cuda_result(raw: i32) -> Result<(), CuResult> {
    if raw == 0 {
        Ok(())
    } else {
        let cr = CuResult::from_raw(raw as u32);
        tracing::warn!(cuda_error = raw, mapped = ?cr, "CUDA call returned error");
        Err(cr)
    }
}

/// Require that a function pointer is present, or return `NotFound`
/// (maps to "feature not supported for this driver version").
fn require_fn<T: Copy>(opt: &Option<T>) -> Result<T, CuResult> {
    opt.ok_or_else(|| {
        tracing::warn!("CUDA function not available (symbol not loaded)");
        CuResult::NotFound
    })
}

// ---------------------------------------------------------------------------
// Kernel parameter deserialization
// ---------------------------------------------------------------------------

/// Deserialize kernel parameters from the wire format into individual param
/// byte buffers.
///
/// Wire format:
/// ```text
/// [4B num_params: u32 LE]
/// [4B param_sizes[0]: u32 LE][param_bytes[0]...]
/// [4B param_sizes[1]: u32 LE][param_bytes[1]...]
/// ...
/// ```
///
/// Returns a `Vec<Vec<u8>>` where each inner `Vec` holds the raw bytes of
/// one kernel parameter.  The caller builds a `void**` array by taking
/// pointers to each `Vec`'s data.
pub(crate) fn deserialize_kernel_params(data: &[u8]) -> Result<Vec<Vec<u8>>, CuResult> {
    if data.len() < 4 {
        return Err(CuResult::InvalidValue);
    }
    let num_params = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    if num_params == 0 {
        return Ok(Vec::new());
    }
    // CUDA limits total kernel param bytes to 4096; cap count to prevent OOM from malformed input
    const MAX_KERNEL_PARAMS: usize = 1024;
    if num_params > MAX_KERNEL_PARAMS {
        return Err(CuResult::InvalidValue);
    }

    let mut offset = 4usize;
    let mut params = Vec::with_capacity(num_params);

    for _ in 0..num_params {
        // Need at least 4 bytes for the size prefix
        if offset + 4 > data.len() {
            return Err(CuResult::InvalidValue);
        }
        let size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        // Zero-size kernel parameters are not valid in CUDA
        if size == 0 {
            return Err(CuResult::InvalidValue);
        }

        // Need `size` bytes of param data
        if offset + size > data.len() {
            return Err(CuResult::InvalidValue);
        }
        params.push(data[offset..offset + size].to_vec());
        offset += size;
    }

    Ok(params)
}

// ---------------------------------------------------------------------------
// CudaGpuBackend
// ---------------------------------------------------------------------------

/// A GPU backend that forwards all operations to the real CUDA Driver API
/// via dynamically loaded function pointers.
///
/// The `_lib` field must outlive `api` because the function pointers point
/// into the loaded library's address space.
pub struct CudaGpuBackend {
    api: CudaApi,
    _lib: Library,
}

impl CudaGpuBackend {
    /// Load the CUDA driver library and resolve function pointers.
    ///
    /// Tries platform-appropriate library names:
    /// - Linux: `libcuda.so`, `libcuda.so.1`
    /// - Windows: `nvcuda.dll`
    ///
    /// Returns an error if no CUDA driver library can be found.
    pub fn new() -> Result<Self, OuterLinkError> {
        let lib = unsafe { Library::new("libcuda.so") }
            .or_else(|_| unsafe { Library::new("libcuda.so.1") })
            .or_else(|_| unsafe { Library::new("nvcuda.dll") })
            .map_err(|e| {
                OuterLinkError::Config(format!("Failed to load CUDA driver library: {e}"))
            })?;

        let api = CudaApi::load(&lib);

        tracing::info!("CUDA driver library loaded successfully");

        Ok(Self { api, _lib: lib })
    }

    /// Check VRAM availability and log warnings if usage is high.
    ///
    /// Returns `(free_bytes, total_bytes)` or an error if cuMemGetInfo
    /// is unavailable.
    fn check_vram(&self) -> Result<(usize, usize), CuResult> {
        let func = require_fn(&self.api.cu_mem_get_info)?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            map_cuda_result(func(&mut free, &mut total))?;
        }
        let used = total.saturating_sub(free);
        let usage_pct = if total > 0 {
            used as f64 / total as f64
        } else {
            0.0
        };
        if usage_pct > VRAM_WARN_THRESHOLD {
            tracing::warn!(
                free_mb = free / (1024 * 1024),
                total_mb = total / (1024 * 1024),
                usage_pct = format!("{:.1}%", usage_pct * 100.0),
                "VRAM usage exceeds {}% threshold",
                (VRAM_WARN_THRESHOLD * 100.0) as u32,
            );
        }
        Ok((free, total))
    }
}

impl GpuBackend for CudaGpuBackend {
    fn init(&self) -> CuResult {
        let func = match require_fn(&self.api.cu_init) {
            Ok(f) => f,
            Err(e) => return e,
        };
        match unsafe { map_cuda_result(func(0)) } {
            Ok(()) => CuResult::Success,
            Err(e) => e,
        }
    }

    fn driver_get_version(&self) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_driver_get_version)?;
        let mut version: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut version))?;
        }
        Ok(version)
    }

    fn device_get_count(&self) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_device_get_count)?;
        let mut count: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut count))?;
        }
        Ok(count)
    }

    fn device_get_name(&self, device: i32) -> Result<String, CuResult> {
        let func = require_fn(&self.api.cu_device_get_name)?;
        let mut buf = vec![0u8; 256];
        // cuDeviceGetName wants a CUdevice, but we need cuDeviceGet first
        // to convert ordinal -> CUdevice handle.  However, since CUDA 4.0
        // the ordinal IS the CUdevice for cuDeviceGetName.
        let cu_dev = self.resolve_device(device)?;
        unsafe {
            map_cuda_result(func(buf.as_mut_ptr(), buf.len() as i32, cu_dev))?;
        }
        // Find the null terminator.
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8(buf[..end].to_vec()).map_err(|_| CuResult::Unknown)
    }

    fn device_get_attribute(&self, attrib: i32, device: i32) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_device_get_attribute)?;
        let cu_dev = self.resolve_device(device)?;
        let mut value: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut value, attrib, cu_dev))?;
        }
        Ok(value)
    }

    fn device_total_mem(&self, device: i32) -> Result<usize, CuResult> {
        let func = require_fn(&self.api.cu_device_total_mem)?;
        let cu_dev = self.resolve_device(device)?;
        let mut bytes: usize = 0;
        unsafe {
            map_cuda_result(func(&mut bytes, cu_dev))?;
        }
        Ok(bytes)
    }

    fn device_get_uuid(&self, device: i32) -> Result<[u8; 16], CuResult> {
        let func = require_fn(&self.api.cu_device_get_uuid)?;
        let cu_dev = self.resolve_device(device)?;
        let mut uuid = CuUuidFfi { bytes: [0u8; 16] };
        unsafe {
            map_cuda_result(func(&mut uuid, cu_dev))?;
        }
        Ok(uuid.bytes)
    }

    fn mem_alloc(&self, size: usize) -> Result<u64, CuResult> {
        if size == 0 {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_mem_alloc)?;

        // VRAM safety check: ensure enough free memory remains after allocation.
        let (free, _total) = self.check_vram()?;
        if free < size + VRAM_SAFETY_MARGIN {
            tracing::warn!(
                requested = size,
                free = free,
                safety_margin = VRAM_SAFETY_MARGIN,
                "Refusing allocation: insufficient VRAM (including safety margin)"
            );
            return Err(CuResult::OutOfMemory);
        }

        let mut dptr: u64 = 0;
        unsafe {
            map_cuda_result(func(&mut dptr, size))?;
        }
        tracing::trace!(ptr = dptr, size, "VRAM allocated");
        Ok(dptr)
    }

    fn mem_free(&self, ptr: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_free)?;
        unsafe { map_cuda_result(func(ptr)) }?;
        tracing::trace!(ptr, "VRAM freed");
        Ok(())
    }

    fn mem_alloc_pitch(&self, width_in_bytes: usize, height: usize, element_size: u32) -> Result<(u64, usize), CuResult> {
        if width_in_bytes == 0 || height == 0 {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_mem_alloc_pitch)?;
        let mut dptr: u64 = 0;
        let mut pitch: usize = 0;
        unsafe {
            map_cuda_result(func(&mut dptr, &mut pitch, width_in_bytes, height, element_size))?;
        }
        tracing::trace!(ptr = dptr, pitch, width_in_bytes, height, "pitched VRAM allocated");
        Ok((dptr, pitch))
    }

    fn memcpy_htod(&self, dst: u64, data: &[u8]) -> CuResult {
        let func = match require_fn(&self.api.cu_memcpy_htod) {
            Ok(f) => f,
            Err(e) => return e,
        };
        match unsafe { map_cuda_result(func(dst, data.as_ptr(), data.len())) } {
            Ok(()) => CuResult::Success,
            Err(e) => e,
        }
    }

    fn memcpy_dtoh(&self, src: u64, size: usize) -> Result<Vec<u8>, CuResult> {
        let func = require_fn(&self.api.cu_memcpy_dtoh)?;
        let mut buf = vec![0u8; size];
        unsafe {
            map_cuda_result(func(buf.as_mut_ptr(), src, size))?;
        }
        Ok(buf)
    }

    fn mem_get_info(&self) -> Result<(usize, usize), CuResult> {
        self.check_vram()
    }

    fn ctx_create(&self, flags: u32, device: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_ctx_create)?;
        let cu_dev = self.resolve_device(device)?;
        let mut ctx: usize = 0;
        unsafe {
            map_cuda_result(func(&mut ctx, flags, cu_dev))?;
        }
        tracing::debug!(ctx, device, "CUDA context created");
        Ok(ctx as u64)
    }

    fn ctx_destroy(&self, ctx: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_ctx_destroy)?;
        unsafe {
            map_cuda_result(func(ctx as usize))?;
        }
        tracing::debug!(ctx, "CUDA context destroyed");
        Ok(())
    }

    fn ctx_exists(&self, ctx: u64) -> bool {
        // Validate by attempting cuCtxGetApiVersion on the handle.
        // If the context is invalid/destroyed, CUDA returns an error.
        let func = match &self.api.cu_ctx_get_api_version {
            Some(f) => *f,
            None => return false,
        };
        let mut version: u32 = 0;
        let result = unsafe { func(ctx as usize, &mut version) };
        result == 0 // CUDA_SUCCESS
    }

    fn ctx_get_device(&self, ctx: u64) -> Result<i32, CuResult> {
        // To get the device for a specific context, we need to set it current
        // first, then call cuCtxGetDevice.  Save/restore is not trivial, but
        // the server typically owns contexts so this is acceptable.
        let set_fn = require_fn(&self.api.cu_ctx_set_current)?;
        let get_fn = require_fn(&self.api.cu_ctx_get_device)?;

        // Set the target context as current.
        unsafe {
            map_cuda_result(set_fn(ctx as usize))?;
        }
        let mut device: i32 = 0;
        unsafe {
            map_cuda_result(get_fn(&mut device))?;
        }
        Ok(device)
    }

    fn ctx_synchronize(&self) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_ctx_synchronize)?;
        unsafe {
            map_cuda_result(func())?;
        }
        Ok(())
    }

    // --- Module operations ---

    fn module_load(&self, path: &str) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_module_load)?;
        let c_path = std::ffi::CString::new(path).map_err(|_| CuResult::InvalidValue)?;
        let mut module: usize = 0;
        unsafe {
            map_cuda_result(func(&mut module, c_path.as_ptr() as *const u8))?;
        }
        tracing::debug!(module, path, "CUDA module loaded from file");
        Ok(module as u64)
    }

    fn module_load_fat_binary(&self, data: &[u8]) -> Result<u64, CuResult> {
        if data.is_empty() {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_module_load_fat_binary)?;
        let mut module: usize = 0;
        unsafe {
            map_cuda_result(func(&mut module, data.as_ptr()))?;
        }
        tracing::debug!(module, data_len = data.len(), "CUDA module loaded from fat binary");
        Ok(module as u64)
    }

    fn module_load_data(&self, data: &[u8]) -> Result<u64, CuResult> {
        if data.is_empty() {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_module_load_data)?;
        let mut module: usize = 0;
        unsafe {
            map_cuda_result(func(&mut module, data.as_ptr()))?;
        }
        tracing::debug!(module, data_len = data.len(), "CUDA module loaded");
        Ok(module as u64)
    }

    fn module_load_data_ex(&self, data: &[u8], options: &[(i32, u64)]) -> Result<u64, CuResult> {
        if data.is_empty() {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_module_load_data_ex)?;
        let mut module: usize = 0;

        // Build the CUjit_option and optionValues arrays for the real CUDA call.
        let num_options = options.len() as u32;
        let mut jit_options: Vec<i32> = options.iter().map(|&(opt, _)| opt).collect();
        // optionValues is an array of void* — we cast each u64 value to a pointer.
        let mut jit_values: Vec<*mut std::ffi::c_void> =
            options.iter().map(|&(_, val)| val as *mut std::ffi::c_void).collect();

        unsafe {
            map_cuda_result(func(
                &mut module,
                data.as_ptr(),
                num_options,
                jit_options.as_mut_ptr(),
                jit_values.as_mut_ptr(),
            ))?;
        }
        tracing::debug!(
            module,
            data_len = data.len(),
            num_options,
            "CUDA module loaded with JIT options"
        );
        Ok(module as u64)
    }

    fn module_unload(&self, module: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_module_unload)?;
        unsafe {
            map_cuda_result(func(module as usize))?;
        }
        tracing::debug!(module, "CUDA module unloaded");
        Ok(())
    }

    fn module_get_function(&self, module: u64, name: &str) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_module_get_function)?;
        let c_name = CString::new(name).map_err(|_| CuResult::InvalidValue)?;
        let mut hfunc: usize = 0;
        unsafe {
            map_cuda_result(func(&mut hfunc, module as usize, c_name.as_ptr() as *const u8))?;
        }
        tracing::debug!(module, name, function = hfunc, "CUDA function resolved");
        Ok(hfunc as u64)
    }

    fn module_get_global(&self, module: u64, name: &str) -> Result<(u64, usize), CuResult> {
        let func = require_fn(&self.api.cu_module_get_global)?;
        let c_name = CString::new(name).map_err(|_| CuResult::InvalidValue)?;
        let mut dptr: u64 = 0;
        let mut size: usize = 0;
        unsafe {
            map_cuda_result(func(
                &mut dptr,
                &mut size,
                module as usize,
                c_name.as_ptr() as *const u8,
            ))?;
        }
        tracing::debug!(module, name, ptr = dptr, size, "CUDA global resolved");
        Ok((dptr, size))
    }

    fn func_get_attribute(&self, attrib: i32, func: u64) -> Result<i32, CuResult> {
        let f = require_fn(&self.api.cu_func_get_attribute)?;
        let mut value: i32 = 0;
        unsafe {
            map_cuda_result(f(&mut value, attrib, func as usize))?;
        }
        tracing::trace!(attrib, func, value, "CUDA func attribute queried");
        Ok(value)
    }

    fn func_set_attribute(&self, func: u64, attrib: i32, value: i32) -> Result<(), CuResult> {
        let f = require_fn(&self.api.cu_func_set_attribute)?;
        unsafe {
            map_cuda_result(f(func as usize, attrib, value))?;
        }
        tracing::trace!(func, attrib, value, "CUDA func attribute set");
        Ok(())
    }

    fn ctx_get_cache_config(&self) -> Result<u32, CuResult> {
        let f = require_fn(&self.api.cu_ctx_get_cache_config)?;
        let mut config: i32 = 0;
        unsafe {
            map_cuda_result(f(&mut config))?;
        }
        Ok(config as u32)
    }

    fn ctx_set_cache_config(&self, config: u32) -> Result<(), CuResult> {
        let f = require_fn(&self.api.cu_ctx_set_cache_config)?;
        unsafe {
            map_cuda_result(f(config as i32))
        }
    }

    fn ctx_get_shared_mem_config(&self) -> Result<u32, CuResult> {
        let f = require_fn(&self.api.cu_ctx_get_shared_mem_config)?;
        let mut config: i32 = 0;
        unsafe {
            map_cuda_result(f(&mut config))?;
        }
        Ok(config as u32)
    }

    fn ctx_set_shared_mem_config(&self, config: u32) -> Result<(), CuResult> {
        let f = require_fn(&self.api.cu_ctx_set_shared_mem_config)?;
        unsafe {
            map_cuda_result(f(config as i32))
        }
    }

    fn func_set_cache_config(&self, func: u64, config: u32) -> Result<(), CuResult> {
        let f = require_fn(&self.api.cu_func_set_cache_config)?;
        unsafe {
            map_cuda_result(f(func as usize, config as i32))
        }
    }

    fn func_set_shared_mem_config(&self, func: u64, config: u32) -> Result<(), CuResult> {
        let f = require_fn(&self.api.cu_func_set_shared_mem_config)?;
        unsafe {
            map_cuda_result(f(func as usize, config as i32))
        }
    }

    fn mem_get_address_range(&self, dptr: u64) -> Result<(u64, usize), CuResult> {
        let f = require_fn(&self.api.cu_mem_get_address_range)?;
        let mut base: u64 = 0;
        let mut size: usize = 0;
        unsafe {
            map_cuda_result(f(&mut base, &mut size, dptr))?;
        }
        tracing::trace!(dptr, base, size, "CUDA mem address range queried");
        Ok((base, size))
    }

    // --- Occupancy operations ---

    fn occupancy_max_active_blocks(
        &self,
        func: u64,
        block_size: i32,
        dynamic_smem_size: u64,
        flags: u32,
    ) -> Result<i32, CuResult> {
        let mut num_blocks: i32 = 0;
        // CUDA defines flags=0 as default behavior, so dispatching to the
        // non-flags variant when flags==0 is semantically equivalent and avoids
        // requiring the WithFlags symbol on older drivers.
        if flags != 0 {
            let f = require_fn(&self.api.cu_occupancy_max_active_blocks_with_flags)?;
            unsafe {
                map_cuda_result(f(
                    &mut num_blocks,
                    func as usize,
                    block_size,
                    dynamic_smem_size as usize,
                    flags,
                ))?;
            }
        } else {
            let f = require_fn(&self.api.cu_occupancy_max_active_blocks)?;
            unsafe {
                map_cuda_result(f(
                    &mut num_blocks,
                    func as usize,
                    block_size,
                    dynamic_smem_size as usize,
                ))?;
            }
        }
        tracing::trace!(func, block_size, dynamic_smem_size, flags, num_blocks, "CUDA occupancy max active blocks");
        Ok(num_blocks)
    }

    fn occupancy_max_potential_block_size(
        &self,
        func: u64,
        dynamic_smem_size: u64,
        block_size_limit: i32,
        flags: u32,
    ) -> Result<(i32, i32), CuResult> {
        let mut min_grid_size: i32 = 0;
        let mut block_size: i32 = 0;
        // CUDA defines flags=0 as default behavior, so dispatching to the
        // non-flags variant when flags==0 is semantically equivalent and avoids
        // requiring the WithFlags symbol on older drivers.
        if flags != 0 {
            let f = require_fn(&self.api.cu_occupancy_max_potential_block_size_with_flags)?;
            unsafe {
                map_cuda_result(f(
                    &mut min_grid_size,
                    &mut block_size,
                    func as usize,
                    0, // NULL callback -- server never has the client's callback
                    dynamic_smem_size as usize,
                    block_size_limit,
                    flags,
                ))?;
            }
        } else {
            let f = require_fn(&self.api.cu_occupancy_max_potential_block_size)?;
            unsafe {
                map_cuda_result(f(
                    &mut min_grid_size,
                    &mut block_size,
                    func as usize,
                    0, // NULL callback
                    dynamic_smem_size as usize,
                    block_size_limit,
                ))?;
            }
        }
        tracing::trace!(func, dynamic_smem_size, block_size_limit, flags, min_grid_size, block_size, "CUDA occupancy max potential block size");
        Ok((min_grid_size, block_size))
    }

    // --- Pointer attribute queries ---

    fn pointer_get_attribute(&self, attribute: i32, ptr: u64) -> Result<u64, CuResult> {
        let f = require_fn(&self.api.cu_pointer_get_attribute)?;
        let mut value: u64 = 0;
        unsafe {
            map_cuda_result(f(
                &mut value as *mut u64 as *mut std::ffi::c_void,
                attribute,
                ptr,
            ))?;
        }
        tracing::trace!(attribute, ptr, value, "CUDA pointer attribute queried");
        Ok(value)
    }

    fn pointer_get_attributes(&self, attributes: &[i32], ptr: u64) -> Result<Vec<u64>, CuResult> {
        // For the real CUDA backend, query each attribute individually.
        // cuPointerGetAttributes has a complex void** interface that varies
        // by attribute type; iterating with cuPointerGetAttribute is simpler
        // and equally correct.
        let mut results = Vec::with_capacity(attributes.len());
        for &attr in attributes {
            results.push(self.pointer_get_attribute(attr, ptr)?);
        }
        Ok(results)
    }

    // --- Stream operations ---

    fn stream_create(&self, flags: u32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_stream_create)?;
        let mut stream: usize = 0;
        unsafe {
            map_cuda_result(func(&mut stream, flags))?;
        }
        tracing::trace!(stream, "CUDA stream created");
        Ok(stream as u64)
    }

    fn stream_destroy(&self, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_stream_destroy)?;
        unsafe {
            map_cuda_result(func(stream as usize))?;
        }
        tracing::trace!(stream, "CUDA stream destroyed");
        Ok(())
    }

    fn stream_synchronize(&self, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_stream_synchronize)?;
        unsafe {
            map_cuda_result(func(stream as usize))?;
        }
        Ok(())
    }

    fn stream_query(&self, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_stream_query)?;
        unsafe {
            map_cuda_result(func(stream as usize))?;
        }
        Ok(())
    }

    fn stream_create_with_priority(&self, flags: u32, priority: i32, _ctx: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_stream_create_with_priority)?;
        let mut stream: usize = 0;
        unsafe {
            map_cuda_result(func(&mut stream, flags, priority))?;
        }
        tracing::trace!(stream, flags, priority, "CUDA stream created with priority");
        Ok(stream as u64)
    }

    fn stream_get_priority(&self, stream: u64) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_stream_get_priority)?;
        let mut priority: i32 = 0;
        unsafe {
            map_cuda_result(func(stream as usize, &mut priority))?;
        }
        Ok(priority)
    }

    fn stream_get_flags(&self, stream: u64) -> Result<u32, CuResult> {
        let func = require_fn(&self.api.cu_stream_get_flags)?;
        let mut flags: u32 = 0;
        unsafe {
            map_cuda_result(func(stream as usize, &mut flags))?;
        }
        Ok(flags)
    }

    fn stream_get_ctx(&self, stream: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_stream_get_ctx)?;
        let mut ctx: usize = 0;
        unsafe {
            map_cuda_result(func(stream as usize, &mut ctx))?;
        }
        Ok(ctx as u64)
    }

    // --- Event operations ---

    fn event_create(&self, flags: u32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_event_create)?;
        let mut event: usize = 0;
        unsafe {
            map_cuda_result(func(&mut event, flags))?;
        }
        tracing::trace!(event, "CUDA event created");
        Ok(event as u64)
    }

    fn event_destroy(&self, event: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_event_destroy)?;
        unsafe {
            map_cuda_result(func(event as usize))?;
        }
        tracing::trace!(event, "CUDA event destroyed");
        Ok(())
    }

    fn event_record(&self, event: u64, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_event_record)?;
        unsafe {
            map_cuda_result(func(event as usize, stream as usize))?;
        }
        Ok(())
    }

    fn event_record_with_flags(&self, event: u64, stream: u64, flags: u32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_event_record_with_flags)?;
        unsafe {
            map_cuda_result(func(event as usize, stream as usize, flags))?;
        }
        Ok(())
    }

    fn event_synchronize(&self, event: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_event_synchronize)?;
        unsafe {
            map_cuda_result(func(event as usize))?;
        }
        Ok(())
    }

    fn event_elapsed_time(&self, start: u64, end: u64) -> Result<f32, CuResult> {
        let func = require_fn(&self.api.cu_event_elapsed_time)?;
        let mut ms: f32 = 0.0;
        unsafe {
            map_cuda_result(func(&mut ms, start as usize, end as usize))?;
        }
        Ok(ms)
    }

    fn event_query(&self, event: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_event_query)?;
        unsafe {
            map_cuda_result(func(event as usize))?;
        }
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64, flags: u32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_stream_wait_event)?;
        unsafe {
            map_cuda_result(func(stream as usize, event as usize, flags))?;
        }
        Ok(())
    }

    // --- Memory: host pinned ---

    fn mem_alloc_host(&self, size: usize) -> Result<u64, CuResult> {
        if size == 0 {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_mem_alloc_host)?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            map_cuda_result(func(&mut ptr, size))?;
        }
        tracing::trace!(ptr = ?ptr, size, "pinned host memory allocated");
        Ok(ptr as u64)
    }

    fn mem_host_alloc(&self, byte_size: usize, flags: u32) -> Result<u64, CuResult> {
        if byte_size == 0 {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_mem_host_alloc)?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            map_cuda_result(func(&mut ptr, byte_size, flags))?;
        }
        tracing::trace!(ptr = ?ptr, byte_size, flags, "host memory allocated with flags");
        Ok(ptr as u64)
    }

    fn mem_free_host(&self, ptr: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_free_host)?;
        unsafe {
            map_cuda_result(func(ptr as *mut std::ffi::c_void))?;
        }
        tracing::trace!(ptr, "pinned host memory freed");
        Ok(())
    }

    fn mem_host_get_device_pointer(&self, host_ptr: u64, flags: u32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_host_get_device_pointer)?;
        let mut dev_ptr: u64 = 0;
        unsafe {
            map_cuda_result(func(&mut dev_ptr, host_ptr as *mut std::ffi::c_void, flags))?;
        }
        Ok(dev_ptr)
    }

    fn mem_host_get_flags(&self, ptr: u64) -> Result<u32, CuResult> {
        let func = require_fn(&self.api.cu_mem_host_get_flags)?;
        let mut flags: u32 = 0;
        unsafe {
            map_cuda_result(func(&mut flags, ptr as *mut std::ffi::c_void))?;
        }
        Ok(flags)
    }

    fn mem_host_register(&self, ptr: u64, size: usize, flags: u32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_host_register)?;
        unsafe {
            map_cuda_result(func(ptr as *mut std::ffi::c_void, size, flags))?;
        }
        Ok(())
    }

    fn mem_host_unregister(&self, ptr: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_host_unregister)?;
        unsafe {
            map_cuda_result(func(ptr as *mut std::ffi::c_void))?;
        }
        Ok(())
    }

    // --- Memory: device-to-device ---

    fn memcpy_dtod(&self, dst: u64, src: u64, size: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memcpy_dtod)?;
        unsafe {
            map_cuda_result(func(dst, src, size as usize))?;
        }
        Ok(())
    }

    fn memcpy(&self, dst: u64, src: u64, size: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memcpy)?;
        unsafe {
            map_cuda_result(func(dst, src, size as usize))?;
        }
        Ok(())
    }

    fn memcpy_async(&self, dst: u64, src: u64, size: u64, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memcpy_async)?;
        unsafe {
            map_cuda_result(func(dst, src, size as usize, stream as usize))?;
        }
        Ok(())
    }

    fn memcpy_dtod_async(&self, dst: u64, src: u64, byte_count: usize, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memcpy_dtod_async)?;
        unsafe {
            map_cuda_result(func(dst, src, byte_count, stream as usize))?;
        }
        Ok(())
    }

    // --- Async memory copy ---

    fn memcpy_htod_async(&self, dst: u64, data: &[u8], stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memcpy_htod_async)?;
        unsafe {
            map_cuda_result(func(dst, data.as_ptr(), data.len(), stream as usize))?;
        }
        Ok(())
    }

    fn memcpy_dtoh_async(&self, src: u64, size: usize, stream: u64) -> Result<Vec<u8>, CuResult> {
        let func = require_fn(&self.api.cu_memcpy_dtoh_async)?;
        let mut buf = vec![0u8; size];
        unsafe {
            map_cuda_result(func(buf.as_mut_ptr(), src, size, stream as usize))?;
        }
        Ok(buf)
    }

    // --- Memset operations ---

    fn memset_d8(&self, dst: u64, value: u8, count: usize) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memset_d8)?;
        unsafe {
            map_cuda_result(func(dst, value, count))?;
        }
        Ok(())
    }

    fn memset_d32(&self, dst: u64, value: u32, count: usize) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memset_d32)?;
        unsafe {
            map_cuda_result(func(dst, value, count))?;
        }
        Ok(())
    }

    fn memset_d8_async(&self, dst: u64, value: u8, count: usize, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memset_d8_async)?;
        unsafe {
            map_cuda_result(func(dst, value, count, stream as usize))?;
        }
        Ok(())
    }

    fn memset_d32_async(&self, dst: u64, value: u32, count: usize, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memset_d32_async)?;
        unsafe {
            map_cuda_result(func(dst, value, count, stream as usize))?;
        }
        Ok(())
    }

    fn memset_d16(&self, dst: u64, value: u16, count: usize) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memset_d16)?;
        unsafe {
            map_cuda_result(func(dst, value, count))?;
        }
        Ok(())
    }

    fn memset_d16_async(&self, dst: u64, value: u16, count: usize, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_memset_d16_async)?;
        unsafe {
            map_cuda_result(func(dst, value, count, stream as usize))?;
        }
        Ok(())
    }

    // --- Kernel launch ---

    fn launch_kernel(
        &self,
        func: u64,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &[u8],
    ) -> Result<(), CuResult> {
        let launch_fn = require_fn(&self.api.cu_launch_kernel)?;

        // Unpack the serialized kernel parameters.
        //
        // Wire format (from the client):
        //   [4B num_params: u32 LE]
        //   For each param:
        //     [4B size: u32 LE][size bytes of raw param data]
        //
        // The client always sends at least the 4-byte num_params field.
        let param_values = deserialize_kernel_params(params)?;

        // Build the void** pointer array that CUDA expects.
        // Each entry points to the start of the corresponding param's raw bytes.
        //
        // SAFETY: param_values is not moved, dropped, or mutated after this point.
        // param_ptrs[i] aliases param_values[i]'s heap buffer, which is stable
        // because we do not push/remove from param_values. Drop order (LIFO) ensures
        // param_ptrs drops before param_values. CUDA driver functions are thread-safe
        // when called with separate contexts.
        let mut param_ptrs: Vec<*mut std::ffi::c_void> = param_values
            .iter()
            .map(|v| v.as_ptr() as *mut std::ffi::c_void)
            .collect();

        // Pass null when there are no kernel arguments; CUDA requires this.
        let params_ptr = if param_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            param_ptrs.as_mut_ptr()
        };

        unsafe {
            map_cuda_result(launch_fn(
                func as usize,
                grid_dim[0],
                grid_dim[1],
                grid_dim[2],
                block_dim[0],
                block_dim[1],
                block_dim[2],
                shared_mem,
                stream as usize,
                params_ptr,
                std::ptr::null_mut(),
            ))?;
        }

        tracing::trace!(
            func,
            grid = ?grid_dim,
            block = ?block_dim,
            shared_mem,
            stream,
            "CUDA kernel launched"
        );
        Ok(())
    }

    fn launch_cooperative_kernel(
        &self,
        func: u64,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &[u8],
    ) -> Result<(), CuResult> {
        let launch_fn = require_fn(&self.api.cu_launch_cooperative_kernel)?;
        let param_values = deserialize_kernel_params(params)?;
        let mut param_ptrs: Vec<*mut std::ffi::c_void> = param_values
            .iter()
            .map(|v| v.as_ptr() as *mut std::ffi::c_void)
            .collect();
        let params_ptr = if param_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            param_ptrs.as_mut_ptr()
        };

        unsafe {
            map_cuda_result(launch_fn(
                func as usize,
                grid_dim[0],
                grid_dim[1],
                grid_dim[2],
                block_dim[0],
                block_dim[1],
                block_dim[2],
                shared_mem,
                stream as usize,
                params_ptr,
            ))?;
        }
        tracing::trace!(func, grid = ?grid_dim, block = ?block_dim, shared_mem, stream, "CUDA cooperative kernel launched");
        Ok(())
    }

    fn device_get_pci_bus_id(&self, device: i32) -> Result<String, CuResult> {
        let func = require_fn(&self.api.cu_device_get_pci_bus_id)?;
        let mut buf = vec![0u8; 64];
        unsafe {
            map_cuda_result(func(buf.as_mut_ptr(), buf.len() as i32, device))?;
        }
        let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let s = String::from_utf8_lossy(&buf[..nul_pos]).into_owned();
        Ok(s)
    }

    fn device_get_by_pci_bus_id(&self, pci_bus_id: &str) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_device_get_by_pci_bus_id)?;
        let c_str = CString::new(pci_bus_id).map_err(|_| CuResult::InvalidValue)?;
        let mut dev: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut dev, c_str.as_ptr() as *const u8))?;
        }
        Ok(dev)
    }

    fn ctx_push_current(&self, ctx: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_ctx_push_current)?;
        unsafe {
            map_cuda_result(func(ctx as usize))
        }
    }

    fn ctx_pop_current(&self) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_ctx_pop_current)?;
        let mut ctx: usize = 0;
        unsafe {
            map_cuda_result(func(&mut ctx))?;
        }
        Ok(ctx as u64)
    }

    fn ctx_get_api_version(&self, ctx: u64) -> Result<u32, CuResult> {
        let func = require_fn(&self.api.cu_ctx_get_api_version)?;
        let mut version: u32 = 0;
        unsafe {
            map_cuda_result(func(ctx as usize, &mut version))?;
        }
        Ok(version)
    }

    fn ctx_get_limit(&self, limit: u32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_ctx_get_limit)?;
        let mut value: u64 = 0;
        unsafe {
            map_cuda_result(func(&mut value, limit as i32))?;
        }
        Ok(value)
    }

    fn ctx_set_limit(&self, limit: u32, value: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_ctx_set_limit)?;
        unsafe {
            map_cuda_result(func(limit as i32, value))
        }
    }

    fn ctx_get_stream_priority_range(&self) -> Result<(i32, i32), CuResult> {
        let func = require_fn(&self.api.cu_ctx_get_stream_priority_range)?;
        let mut least: i32 = 0;
        let mut greatest: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut least, &mut greatest))?;
        }
        Ok((least, greatest))
    }

    fn ctx_get_flags(&self, _ctx: u64) -> Result<u32, CuResult> {
        let func = require_fn(&self.api.cu_ctx_get_flags)?;
        let mut flags: u32 = 0;
        unsafe {
            map_cuda_result(func(&mut flags))?;
        }
        Ok(flags)
    }

    fn device_can_access_peer(&self, dev: i32, peer_dev: i32) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_device_can_access_peer)?;
        let dev_h = self.resolve_device(dev)?;
        let peer_h = self.resolve_device(peer_dev)?;
        let mut can_access: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut can_access, dev_h, peer_h))?;
        }
        Ok(can_access)
    }

    fn device_get_p2p_attribute(&self, attrib: i32, src_device: i32, dst_device: i32) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_device_get_p2p_attribute)?;
        let src = self.resolve_device(src_device)?;
        let dst = self.resolve_device(dst_device)?;
        let mut value: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut value, attrib, src, dst))?;
        }
        Ok(value)
    }

    fn ctx_enable_peer_access(&self, peer_ctx: u64, flags: u32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_ctx_enable_peer_access)?;
        unsafe {
            map_cuda_result(func(peer_ctx as usize, flags))
        }
    }

    fn ctx_disable_peer_access(&self, peer_ctx: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_ctx_disable_peer_access)?;
        unsafe {
            map_cuda_result(func(peer_ctx as usize))
        }
    }

    fn link_create(&self, _options: &[(i32, u64)]) -> Result<u64, CuResult> {
        // TODO: Wire to real cuLinkCreate_v2 via libloading.
        tracing::warn!("CudaBackend::link_create not yet wired to real CUDA driver");
        Err(CuResult::JitCompilerNotFound)
    }

    fn link_add_data(&self, _state: u64, _jit_type: i32, _data: &[u8], _name: &str, _options: &[(i32, u64)]) -> Result<(), CuResult> {
        tracing::warn!("CudaBackend::link_add_data not yet wired to real CUDA driver");
        Err(CuResult::JitCompilerNotFound)
    }

    fn link_add_file(&self, _state: u64, _jit_type: i32, _path: &str, _options: &[(i32, u64)]) -> Result<(), CuResult> {
        tracing::warn!("CudaBackend::link_add_file not yet wired to real CUDA driver");
        Err(CuResult::JitCompilerNotFound)
    }

    fn link_complete(&self, _state: u64) -> Result<(u64, Vec<u8>), CuResult> {
        tracing::warn!("CudaBackend::link_complete not yet wired to real CUDA driver");
        Err(CuResult::JitCompilerNotFound)
    }

    fn link_destroy(&self, _state: u64) -> Result<(), CuResult> {
        tracing::warn!("CudaBackend::link_destroy not yet wired to real CUDA driver");
        Err(CuResult::JitCompilerNotFound)
    }

    fn stream_add_callback(&self, _stream: u64, _callback_id: u64, _flags: u32) -> Result<(), CuResult> {
        // In the real backend, we would use cuLaunchHostFunc to enqueue a
        // notification-sender on the real GPU stream. For now, callbacks fire
        // immediately (the handler sends CallbackReady right after the response).
        Ok(())
    }

    fn launch_host_func(&self, _stream: u64, _callback_id: u64) -> Result<(), CuResult> {
        // Same as stream_add_callback -- immediate notification in Phase 1.
        Ok(())
    }

    // --- Library API (CUDA 12+) ---

    fn library_load_data(
        &self,
        image: &[u8],
        jit_options: &[(i32, u64)],
        lib_options: &[(i32, u64)],
    ) -> Result<u64, CuResult> {
        if image.is_empty() {
            return Err(CuResult::InvalidValue);
        }
        let func = require_fn(&self.api.cu_library_load_data)?;
        let mut library: usize = 0;

        // Build JIT option arrays (same pattern as module_load_data_ex).
        let jit_opt_keys: Vec<i32> = jit_options.iter().map(|(k, _)| *k).collect();
        let mut jit_opt_vals: Vec<*mut std::ffi::c_void> =
            jit_options.iter().map(|(_, v)| *v as *mut std::ffi::c_void).collect();

        // Build lib option arrays.
        let lib_opt_keys: Vec<i32> = lib_options.iter().map(|(k, _)| *k).collect();
        let mut lib_opt_vals: Vec<*mut std::ffi::c_void> =
            lib_options.iter().map(|(_, v)| *v as *mut std::ffi::c_void).collect();

        unsafe {
            map_cuda_result(func(
                &mut library,
                image.as_ptr(),
                jit_opt_keys.as_ptr(),
                jit_opt_vals.as_mut_ptr(),
                jit_options.len() as u32,
                lib_opt_keys.as_ptr(),
                lib_opt_vals.as_mut_ptr(),
                lib_options.len() as u32,
            ))?;
        }
        Ok(library as u64)
    }

    fn library_unload(&self, library: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_library_unload)?;
        unsafe {
            map_cuda_result(func(library as usize))?;
        }
        Ok(())
    }

    fn library_get_kernel(&self, library: u64, name: &str) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_library_get_kernel)?;
        let c_name = CString::new(name).map_err(|_| CuResult::InvalidValue)?;
        let mut kernel: usize = 0;
        unsafe {
            map_cuda_result(func(&mut kernel, library as usize, c_name.as_ptr() as *const u8))?;
        }
        Ok(kernel as u64)
    }

    fn library_get_module(&self, library: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_library_get_module)?;
        let mut module: usize = 0;
        unsafe {
            map_cuda_result(func(&mut module, library as usize))?;
        }
        Ok(module as u64)
    }

    fn kernel_get_function(&self, kernel: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_kernel_get_function)?;
        let mut function: usize = 0;
        unsafe {
            map_cuda_result(func(&mut function, kernel as usize))?;
        }
        Ok(function as u64)
    }

    fn stream_begin_capture(&self, stream: u64, mode: i32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_stream_begin_capture)?;
        unsafe {
            map_cuda_result(func(stream as usize, mode))?;
        }
        Ok(())
    }

    fn stream_end_capture(&self, stream: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_stream_end_capture)?;
        let mut graph: usize = 0;
        unsafe {
            map_cuda_result(func(stream as usize, &mut graph))?;
        }
        Ok(graph as u64)
    }

    fn stream_is_capturing(&self, stream: u64) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_stream_is_capturing)?;
        let mut status: i32 = 0;
        unsafe {
            map_cuda_result(func(stream as usize, &mut status))?;
        }
        Ok(status)
    }

    fn stream_get_capture_info(&self, stream: u64) -> Result<(i32, u64), CuResult> {
        let func = require_fn(&self.api.cu_stream_get_capture_info)?;
        let mut status: i32 = 0;
        let mut id: u64 = 0;
        unsafe {
            map_cuda_result(func(
                stream as usize,
                &mut status,
                &mut id,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ))?;
        }
        Ok((status, id))
    }

    fn graph_create(&self, flags: u32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_graph_create)?;
        let mut graph: usize = 0;
        unsafe {
            map_cuda_result(func(&mut graph, flags))?;
        }
        Ok(graph as u64)
    }

    fn graph_instantiate(&self, graph: u64, flags: u64) -> Result<u64, CuResult> {
        let mut exec: usize = 0;
        // Try cuGraphInstantiateWithFlags first (CUDA 11.4+), fall back to v2.
        if flags != 0 {
            if let Ok(func) = require_fn(&self.api.cu_graph_instantiate_with_flags) {
                unsafe {
                    map_cuda_result(func(&mut exec, graph as usize, flags))?;
                }
                return Ok(exec as u64);
            }
        }
        let func = require_fn(&self.api.cu_graph_instantiate)?;
        unsafe {
            map_cuda_result(func(
                &mut exec,
                graph as usize,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ))?;
        }
        Ok(exec as u64)
    }

    fn graph_launch(&self, graph_exec: u64, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_graph_launch)?;
        unsafe {
            map_cuda_result(func(graph_exec as usize, stream as usize))?;
        }
        Ok(())
    }

    fn graph_exec_destroy(&self, graph_exec: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_graph_exec_destroy)?;
        unsafe {
            map_cuda_result(func(graph_exec as usize))?;
        }
        Ok(())
    }

    fn graph_destroy(&self, graph: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_graph_destroy)?;
        unsafe {
            map_cuda_result(func(graph as usize))?;
        }
        Ok(())
    }

    fn shutdown(&self) {
        // The CUDA driver cleans up all resources when the process exits,
        // but we log explicitly so operators know cleanup was intentional.
        tracing::info!("CUDA backend: shutdown requested, driver will reclaim resources on exit");
    }

    fn primary_ctx_retain(&self, device: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_device_primary_ctx_retain)?;
        let dev = self.resolve_device(device)?;
        let mut ctx: usize = 0;
        unsafe {
            map_cuda_result(func(&mut ctx, dev))?;
        }
        Ok(ctx as u64)
    }

    fn primary_ctx_release(&self, device: i32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_device_primary_ctx_release)?;
        let dev = self.resolve_device(device)?;
        unsafe {
            map_cuda_result(func(dev))
        }
    }

    fn primary_ctx_get_state(&self, device: i32) -> Result<(u32, i32), CuResult> {
        let func = require_fn(&self.api.cu_device_primary_ctx_get_state)?;
        let dev = self.resolve_device(device)?;
        let mut flags: u32 = 0;
        let mut active: i32 = 0;
        unsafe {
            map_cuda_result(func(dev, &mut flags, &mut active))?;
        }
        Ok((flags, active))
    }

    fn primary_ctx_set_flags(&self, device: i32, flags: u32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_device_primary_ctx_set_flags)?;
        let dev = self.resolve_device(device)?;
        unsafe {
            map_cuda_result(func(dev, flags))
        }
    }

    fn primary_ctx_reset(&self, device: i32) -> Result<Option<u64>, CuResult> {
        let func = require_fn(&self.api.cu_device_primary_ctx_reset)?;
        let dev = self.resolve_device(device)?;
        unsafe {
            map_cuda_result(func(dev))?;
        }
        // Real CUDA doesn't return the old handle; we return None.
        Ok(None)
    }

    // --- Stream-ordered memory / pool operations ---

    fn mem_alloc_async(&self, size: usize, stream: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_alloc_async)?;
        let mut dptr: u64 = 0;
        unsafe {
            map_cuda_result(func(&mut dptr, size, stream as usize))?;
        }
        Ok(dptr)
    }

    fn mem_free_async(&self, ptr: u64, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_free_async)?;
        unsafe {
            map_cuda_result(func(ptr, stream as usize))?;
        }
        Ok(())
    }

    fn device_get_default_mem_pool(&self, device: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_device_get_default_mem_pool)?;
        let dev = self.resolve_device(device)?;
        let mut pool: usize = 0;
        unsafe {
            map_cuda_result(func(&mut pool, dev))?;
        }
        Ok(pool as u64)
    }

    fn mem_pool_create(&self, alloc_type: i32, location_type: i32, location_id: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_pool_create)?;
        let props = CuMemPoolProps {
            alloc_type,
            handle_types: 0, // CU_MEM_HANDLE_TYPE_NONE
            location_type,
            location_id,
            win32_security: 0,
            max_size: 0,
            reserved: [0u8; 56],
        };
        let mut pool: usize = 0;
        unsafe {
            map_cuda_result(func(&mut pool, &props))?;
        }
        Ok(pool as u64)
    }

    fn mem_pool_destroy(&self, pool: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_pool_destroy)?;
        unsafe {
            map_cuda_result(func(pool as usize))?;
        }
        Ok(())
    }

    fn mem_pool_get_attribute(&self, pool: u64, attr: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_pool_get_attribute)?;
        let mut value: u64 = 0;
        unsafe {
            map_cuda_result(func(pool as usize, attr, &mut value))?;
        }
        Ok(value)
    }

    fn mem_pool_set_attribute(&self, pool: u64, attr: i32, value: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_pool_set_attribute)?;
        unsafe {
            map_cuda_result(func(pool as usize, attr, &value))?;
        }
        Ok(())
    }

    fn mem_pool_trim_to(&self, pool: u64, min_bytes: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_pool_trim_to)?;
        unsafe {
            map_cuda_result(func(pool as usize, min_bytes as usize))?;
        }
        Ok(())
    }

    fn mem_alloc_from_pool_async(&self, size: usize, pool: u64, stream: u64) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_alloc_from_pool_async)?;
        let mut dptr: u64 = 0;
        unsafe {
            map_cuda_result(func(&mut dptr, size, pool as usize, stream as usize))?;
        }
        Ok(dptr)
    }

    fn device_get_mem_pool(&self, device: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_device_get_mem_pool)?;
        let mut pool: usize = 0;
        unsafe {
            map_cuda_result(func(&mut pool, device))?;
        }
        Ok(pool as u64)
    }

    fn device_set_mem_pool(&self, device: i32, pool: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_device_set_mem_pool)?;
        unsafe {
            map_cuda_result(func(device, pool as usize))?;
        }
        Ok(())
    }

    fn mem_get_allocation_granularity(&self, location_type: i32, location_id: i32, option: i32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_get_allocation_granularity)?;
        #[repr(C)]
        struct CuMemLocation {
            loc_type: i32,
            id: i32,
        }
        let loc = CuMemLocation { loc_type: location_type, id: location_id };
        let mut granularity: usize = 0;
        unsafe {
            map_cuda_result(func(&mut granularity, &loc as *const _ as *const std::ffi::c_void, option))?;
        }
        Ok(granularity as u64)
    }

    // --- Managed / unified memory ---

    fn mem_alloc_managed(&self, byte_size: usize, flags: u32) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_alloc_managed)?;
        if byte_size == 0 {
            return Err(CuResult::InvalidValue);
        }
        let mut dptr: u64 = 0;
        unsafe {
            map_cuda_result(func(&mut dptr, byte_size, flags))?;
        }
        Ok(dptr)
    }

    fn mem_prefetch_async(&self, dptr: u64, count: usize, dst_device: i32, stream: u64) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_prefetch_async)?;
        unsafe {
            map_cuda_result(func(dptr, count, dst_device, stream as usize))?;
        }
        Ok(())
    }

    fn mem_advise(&self, dptr: u64, count: usize, advice: i32, device: i32) -> Result<(), CuResult> {
        let func = require_fn(&self.api.cu_mem_advise)?;
        unsafe {
            map_cuda_result(func(dptr, count, advice, device))?;
        }
        Ok(())
    }

    fn mem_range_get_attribute(&self, attribute: i32, dptr: u64, count: usize) -> Result<u64, CuResult> {
        let func = require_fn(&self.api.cu_mem_range_get_attribute)?;
        let mut value: u64 = 0;
        unsafe {
            map_cuda_result(func(
                &mut value as *mut u64 as *mut std::ffi::c_void,
                std::mem::size_of::<u64>(),
                attribute,
                dptr,
                count,
            ))?;
        }
        Ok(value)
    }
}

impl CudaGpuBackend {
    /// Convert a device ordinal to a CUdevice handle via cuDeviceGet.
    fn resolve_device(&self, ordinal: i32) -> Result<i32, CuResult> {
        let func = require_fn(&self.api.cu_device_get)?;
        let mut dev: i32 = 0;
        unsafe {
            map_cuda_result(func(&mut dev, ordinal))?;
        }
        Ok(dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_cuda_result_success() {
        assert_eq!(map_cuda_result(0), Ok(()));
    }

    #[test]
    fn map_cuda_result_error() {
        let err = map_cuda_result(2).unwrap_err();
        assert_eq!(err, CuResult::OutOfMemory);
    }

    #[test]
    fn map_cuda_result_unknown() {
        let err = map_cuda_result(99999).unwrap_err();
        assert_eq!(err, CuResult::Unknown);
    }

    #[test]
    fn require_fn_some() {
        let f: Option<FnCuInit> = Some(dummy_cu_init);
        assert!(require_fn(&f).is_ok());
    }

    #[test]
    fn require_fn_none() {
        let f: Option<FnCuInit> = None;
        assert_eq!(require_fn(&f).unwrap_err(), CuResult::NotFound);
    }

    #[test]
    fn vram_safety_margin_value() {
        assert_eq!(VRAM_SAFETY_MARGIN, 512 * 1024 * 1024);
    }

    // Dummy for require_fn test.
    unsafe extern "C" fn dummy_cu_init(_flags: u32) -> i32 {
        0
    }

    // --- deserialize_kernel_params tests ---

    #[test]
    fn deserialize_kernel_params_empty() {
        // num_params = 0
        let data = 0u32.to_le_bytes();
        let result = deserialize_kernel_params(&data).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn deserialize_kernel_params_single_u64() {
        // One param: a u64 value
        let val: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let val_bytes = val.to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes()); // num_params = 1
        data.extend_from_slice(&8u32.to_le_bytes()); // param_sizes[0] = 8
        data.extend_from_slice(&val_bytes);           // param_bytes[0]
        let result = deserialize_kernel_params(&data).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], val_bytes);
    }

    #[test]
    fn deserialize_kernel_params_two_params() {
        // Two params: a u64 (device pointer) and a u32 (count)
        let ptr_val: u64 = 0x1234_5678_ABCD_0000;
        let count_val: u32 = 1024;
        let mut data = Vec::new();
        data.extend_from_slice(&2u32.to_le_bytes()); // num_params = 2
        data.extend_from_slice(&8u32.to_le_bytes()); // param_sizes[0] = 8
        data.extend_from_slice(&ptr_val.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // param_sizes[1] = 4
        data.extend_from_slice(&count_val.to_le_bytes());
        let result = deserialize_kernel_params(&data).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ptr_val.to_le_bytes());
        assert_eq!(result[1], count_val.to_le_bytes());
    }

    #[test]
    fn deserialize_kernel_params_three_mixed_sizes() {
        // Three params: u64, f32, u32
        let a: u64 = 0xFF00_FF00_FF00_FF00;
        let b: f32 = 3.14;
        let c: u32 = 42;
        let mut data = Vec::new();
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&a.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&b.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&c.to_le_bytes());
        let result = deserialize_kernel_params(&data).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(u64::from_le_bytes(result[0].clone().try_into().unwrap()), a);
        assert_eq!(f32::from_le_bytes(result[1].clone().try_into().unwrap()), b);
        assert_eq!(u32::from_le_bytes(result[2].clone().try_into().unwrap()), c);
    }

    #[test]
    fn deserialize_kernel_params_too_short() {
        // Less than 4 bytes -- can't even read num_params
        let data = [0u8; 2];
        assert_eq!(
            deserialize_kernel_params(&data).unwrap_err(),
            CuResult::InvalidValue
        );
    }

    #[test]
    fn deserialize_kernel_params_truncated_size() {
        // Says 1 param but no size prefix
        let data = 1u32.to_le_bytes();
        assert_eq!(
            deserialize_kernel_params(&data).unwrap_err(),
            CuResult::InvalidValue
        );
    }

    #[test]
    fn deserialize_kernel_params_truncated_data() {
        // Says 1 param, size = 8, but only 4 bytes of data
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 4]); // only 4 of 8 bytes
        assert_eq!(
            deserialize_kernel_params(&data).unwrap_err(),
            CuResult::InvalidValue
        );
    }
}
