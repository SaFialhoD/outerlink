//! GPU backend trait and stub implementation.
//!
//! The `GpuBackend` trait abstracts all GPU operations so the server can
//! run against a real CUDA driver **or** a stub that fakes everything in
//! memory. The stub is used for testing, CI, and development on machines
//! without a GPU.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use outerlink_common::cuda_types::CuResult;

// ---------------------------------------------------------------------------
// Occupancy stub constants (Ampere / GA102 defaults)
// ---------------------------------------------------------------------------

/// Number of SMs on the stub GPU (RTX 3090 = 82 SMs).
const STUB_NUM_SMS: i32 = 82;
/// Maximum resident threads per SM on Ampere.
const STUB_MAX_THREADS_PER_SM: i32 = 2048;
/// Maximum resident blocks per SM on Ampere.
const STUB_MAX_BLOCKS_PER_SM: i32 = 16;

/// Trait abstracting GPU operations.
///
/// Real implementations load `libcuda.so` / `nvcuda.dll` via `libloading`
/// and forward every call to the driver. The [`StubGpuBackend`] keeps
/// everything in-process for testing.
pub trait GpuBackend: Send + Sync {
    /// Initialise the CUDA driver (corresponds to `cuInit`).
    fn init(&self) -> CuResult;

    /// Return the driver version (corresponds to `cuDriverGetVersion`).
    fn driver_get_version(&self) -> Result<i32, CuResult>;

    /// Return the number of CUDA-capable devices.
    fn device_get_count(&self) -> Result<i32, CuResult>;

    /// Return a human-readable device name.
    fn device_get_name(&self, device: i32) -> Result<String, CuResult>;

    /// Query a device attribute by its raw integer code.
    fn device_get_attribute(&self, attrib: i32, device: i32) -> Result<i32, CuResult>;

    /// Return total memory in bytes for the given device.
    fn device_total_mem(&self, device: i32) -> Result<usize, CuResult>;

    /// Return the 16-byte UUID of a device.
    fn device_get_uuid(&self, device: i32) -> Result<[u8; 16], CuResult>;

    /// Allocate `size` bytes of device memory and return a device pointer.
    fn mem_alloc(&self, size: usize) -> Result<u64, CuResult>;

    /// Free a previously-allocated device pointer.
    fn mem_free(&self, ptr: u64) -> Result<(), CuResult>;

    /// Copy host data to device memory (host-to-device).
    fn memcpy_htod(&self, dst: u64, data: &[u8]) -> CuResult;

    /// Copy `size` bytes from device memory to a host buffer (device-to-host).
    fn memcpy_dtoh(&self, src: u64, size: usize) -> Result<Vec<u8>, CuResult>;

    /// Return `(free, total)` memory in bytes.
    fn mem_get_info(&self) -> Result<(usize, usize), CuResult>;

    /// Create a new CUDA context on the given device.
    fn ctx_create(&self, flags: u32, device: i32) -> Result<u64, CuResult>;

    /// Destroy a CUDA context.
    fn ctx_destroy(&self, ctx: u64) -> Result<(), CuResult>;

    /// Check whether a context handle exists (is valid).
    ///
    /// Used by the session layer to validate `cuCtxSetCurrent` calls
    /// without the backend needing to track per-connection state.
    fn ctx_exists(&self, ctx: u64) -> bool;

    /// Get the device ordinal for a context.
    fn ctx_get_device(&self, ctx: u64) -> Result<i32, CuResult>;

    /// Synchronize the current context (block until all work completes).
    fn ctx_synchronize(&self) -> Result<(), CuResult>;

    // --- Module operations ---

    /// Load a module from raw data (e.g. PTX or cubin).
    fn module_load_data(&self, data: &[u8]) -> Result<u64, CuResult>;

    /// Load a module from raw data with JIT compiler options.
    ///
    /// Options are `(CUjit_option, value)` pairs. In stub mode the options
    /// are ignored and this delegates to [`module_load_data`].
    fn module_load_data_ex(&self, data: &[u8], options: &[(i32, u64)]) -> Result<u64, CuResult>;

    /// Unload a previously loaded module.
    fn module_unload(&self, module: u64) -> Result<(), CuResult>;

    /// Get a function handle from a loaded module by name.
    fn module_get_function(&self, module: u64, name: &str) -> Result<u64, CuResult>;

    /// Get a global variable from a loaded module by name.
    /// Returns `(device_pointer, size_in_bytes)`.
    fn module_get_global(&self, module: u64, name: &str) -> Result<(u64, usize), CuResult>;

    /// Query an integer attribute of a kernel function.
    ///
    /// `attrib` is the raw `CUfunction_attribute` enum value.
    /// Returns the attribute value on success.
    fn func_get_attribute(&self, attrib: i32, func: u64) -> Result<i32, CuResult>;

    /// Set an attribute of a kernel function.
    ///
    /// `attrib` is the raw `CUfunction_attribute` enum value.
    /// The most important attribute is `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` (8).
    fn func_set_attribute(&self, func: u64, attrib: i32, value: i32) -> Result<(), CuResult>;

    /// Get the base address and size of the allocation that contains `dptr`.
    ///
    /// Returns `(base, size)` on success.
    fn mem_get_address_range(&self, dptr: u64) -> Result<(u64, usize), CuResult>;

    // --- Occupancy operations ---

    /// Compute max active blocks per SM for a given function and block size.
    ///
    /// `flags` is passed for the WithFlags variant (0 for the plain variant).
    fn occupancy_max_active_blocks(
        &self,
        func: u64,
        block_size: i32,
        dynamic_smem_size: u64,
        flags: u32,
    ) -> Result<i32, CuResult>;

    /// Compute optimal block size and minimum grid size for a function.
    ///
    /// `flags` is passed for the WithFlags variant (0 for the plain variant).
    /// Returns `(min_grid_size, block_size)`.
    fn occupancy_max_potential_block_size(
        &self,
        func: u64,
        dynamic_smem_size: u64,
        block_size_limit: i32,
        flags: u32,
    ) -> Result<(i32, i32), CuResult>;

    // --- Stream operations ---

    /// Create a new CUDA stream with the given flags.
    fn stream_create(&self, flags: u32) -> Result<u64, CuResult>;

    /// Destroy a CUDA stream.
    fn stream_destroy(&self, stream: u64) -> Result<(), CuResult>;

    /// Block until all work on the stream completes.
    fn stream_synchronize(&self, stream: u64) -> Result<(), CuResult>;

    /// Query whether all work on the stream has completed.
    /// `Ok(())` = complete, `Err(NotReady)` = still busy.
    fn stream_query(&self, stream: u64) -> Result<(), CuResult>;

    /// Create a new CUDA stream with the given flags, priority, and owning context.
    fn stream_create_with_priority(&self, flags: u32, priority: i32, ctx: u64) -> Result<u64, CuResult>;

    /// Get the priority of a stream.
    fn stream_get_priority(&self, stream: u64) -> Result<i32, CuResult>;

    /// Get the creation flags of a stream.
    fn stream_get_flags(&self, stream: u64) -> Result<u32, CuResult>;

    /// Get the context associated with a stream.
    fn stream_get_ctx(&self, stream: u64) -> Result<u64, CuResult>;

    // --- Event operations ---

    /// Create a new CUDA event with the given flags.
    fn event_create(&self, flags: u32) -> Result<u64, CuResult>;

    /// Destroy a CUDA event.
    fn event_destroy(&self, event: u64) -> Result<(), CuResult>;

    /// Record an event on a stream.
    fn event_record(&self, event: u64, stream: u64) -> Result<(), CuResult>;

    /// Block until an event has been recorded (completed).
    fn event_synchronize(&self, event: u64) -> Result<(), CuResult>;

    /// Compute elapsed time in milliseconds between two recorded events.
    fn event_elapsed_time(&self, start: u64, end: u64) -> Result<f32, CuResult>;

    /// Query whether an event has been recorded.
    /// `Ok(())` = recorded, `Err(NotReady)` = not yet.
    fn event_query(&self, event: u64) -> Result<(), CuResult>;

    /// Make a stream wait on an event before executing further work.
    fn stream_wait_event(&self, stream: u64, event: u64, flags: u32) -> Result<(), CuResult>;

    // --- Memory: host pinned ---

    /// Allocate pinned (page-locked) host memory.
    fn mem_alloc_host(&self, size: usize) -> Result<u64, CuResult>;

    /// Free pinned host memory.
    fn mem_free_host(&self, ptr: u64) -> Result<(), CuResult>;

    // --- Memory: async copy ---

    /// Async host-to-device copy (takes stream parameter).
    /// In the stub, behaves the same as sync memcpy_htod.
    fn memcpy_htod_async(&self, dst: u64, data: &[u8], stream: u64) -> Result<(), CuResult>;

    /// Async device-to-host copy (takes stream parameter).
    /// In the stub, behaves the same as sync memcpy_dtoh.
    fn memcpy_dtoh_async(&self, src: u64, size: usize, stream: u64) -> Result<Vec<u8>, CuResult>;

    // --- Memory: memset ---

    /// Fill device memory with a u8 value.
    fn memset_d8(&self, dst: u64, value: u8, count: usize) -> Result<(), CuResult>;

    /// Fill device memory with a u32 value.
    fn memset_d32(&self, dst: u64, value: u32, count: usize) -> Result<(), CuResult>;

    /// Async fill device memory with a u8 value.
    fn memset_d8_async(&self, dst: u64, value: u8, count: usize, stream: u64) -> Result<(), CuResult>;

    /// Async fill device memory with a u32 value.
    fn memset_d32_async(&self, dst: u64, value: u32, count: usize, stream: u64) -> Result<(), CuResult>;

    // --- Memory: device-to-device ---

    /// Copy `size` bytes from one device pointer to another.
    fn memcpy_dtod(&self, dst: u64, src: u64, size: u64) -> Result<(), CuResult>;

    // --- Kernel launch ---

    /// Launch a kernel on a stream.
    fn launch_kernel(
        &self,
        func: u64,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &[u8],
    ) -> Result<(), CuResult>;

    // --- Peer access ---

    /// Check if one device can access another's memory via P2P.
    /// Returns `Ok(1)` if access is possible, `Ok(0)` otherwise.
    fn device_can_access_peer(&self, dev: i32, peer_dev: i32) -> Result<i32, CuResult>;

    /// Get a P2P attribute between two devices.
    /// Returns the attribute value on success.
    fn device_get_p2p_attribute(&self, attrib: i32, src_device: i32, dst_device: i32) -> Result<i32, CuResult>;

    /// Enable peer access from the current context to a peer context.
    /// `flags` is reserved (must be 0 in current CUDA).
    fn ctx_enable_peer_access(&self, peer_ctx: u64, flags: u32) -> Result<(), CuResult>;

    /// Disable peer access previously enabled via `ctx_enable_peer_access`.
    fn ctx_disable_peer_access(&self, peer_ctx: u64) -> Result<(), CuResult>;

    // --- Pointer attribute queries ---

    /// Get a single attribute of a device or host pointer.
    ///
    /// `attribute` is the raw `CUpointer_attribute` value:
    ///   1=CONTEXT, 2=MEMORY_TYPE, 3=DEVICE_POINTER, 4=HOST_POINTER,
    ///   6=IS_MANAGED, 8=DEVICE_ORDINAL.
    /// Returns the attribute value as u64 on success.
    fn pointer_get_attribute(&self, attribute: i32, ptr: u64) -> Result<u64, CuResult>;

    /// Get multiple attributes of a pointer in one call.
    ///
    /// Returns a Vec of u64 values, one per requested attribute, in order.
    fn pointer_get_attributes(&self, attributes: &[i32], ptr: u64) -> Result<Vec<u64>, CuResult>;

    // --- Lifecycle ---

    // --- Context stack & query operations ---

    /// Push a context onto the per-thread context stack.
    fn ctx_push_current(&self, ctx: u64) -> Result<(), CuResult>;

    /// Pop the top context from the stack and return its handle.
    fn ctx_pop_current(&self) -> Result<u64, CuResult>;

    /// Get the API version for a context.
    fn ctx_get_api_version(&self, ctx: u64) -> Result<u32, CuResult>;

    /// Get a resource limit value (stack size, printf FIFO, malloc heap, etc.).
    fn ctx_get_limit(&self, limit: u32) -> Result<u64, CuResult>;

    /// Set a resource limit value.
    fn ctx_set_limit(&self, limit: u32, value: u64) -> Result<(), CuResult>;

    /// Get the range of valid stream priorities (least, greatest).
    fn ctx_get_stream_priority_range(&self) -> Result<(i32, i32), CuResult>;

    /// Get the flags of the current context.
    fn ctx_get_flags(&self, ctx: u64) -> Result<u32, CuResult>;

    /// Get the preferred cache configuration for the current context.
    fn ctx_get_cache_config(&self) -> Result<u32, CuResult>;

    /// Set the preferred cache configuration for the current context.
    fn ctx_set_cache_config(&self, config: u32) -> Result<(), CuResult>;

    /// Get the shared memory configuration (bank size) for the current context.
    fn ctx_get_shared_mem_config(&self) -> Result<u32, CuResult>;

    /// Set the shared memory configuration (bank size) for the current context.
    fn ctx_set_shared_mem_config(&self, config: u32) -> Result<(), CuResult>;

    /// Set the preferred cache configuration for a kernel function.
    fn func_set_cache_config(&self, func: u64, config: u32) -> Result<(), CuResult>;

    /// Set the shared memory configuration for a kernel function.
    fn func_set_shared_mem_config(&self, func: u64, config: u32) -> Result<(), CuResult>;

    // --- Lifecycle ---

    /// Clean up all GPU resources held by this backend.
    ///
    /// Called during graceful shutdown. Implementations should destroy all
    /// contexts, free all allocations, unload modules, destroy streams and
    /// events, etc. After this call, the backend is in a "shut down" state
    /// and should not be used for new operations.
    ///
    /// The default implementation is a no-op, suitable for backends that
    /// do not track resources (e.g. forwarding proxies).
    fn shutdown(&self) {}

    // --- Primary context management ---

    /// Retain a reference to the primary context for `device`.
    /// Increments the refcount and creates the context on first call.
    /// Returns the same handle every time for the same device.
    fn primary_ctx_retain(&self, device: i32) -> Result<u64, CuResult>;

    /// Release a reference to the primary context for `device`.
    /// Decrements the refcount and destroys the context when it reaches 0.
    fn primary_ctx_release(&self, device: i32) -> Result<(), CuResult>;

    /// Get the state of the primary context for `device`.
    /// Returns (flags, active) where active is 1 if refcount > 0.
    fn primary_ctx_get_state(&self, device: i32) -> Result<(u32, i32), CuResult>;

    /// Set flags for the primary context of `device`.
    /// Only valid when the refcount is 0 (context not active).
    fn primary_ctx_set_flags(&self, device: i32, flags: u32) -> Result<(), CuResult>;

    /// Reset the primary context for `device`.
    /// Destroys regardless of refcount, resets refcount to 0.
    /// Returns the old context handle if one existed.
    fn primary_ctx_reset(&self, device: i32) -> Result<Option<u64>, CuResult>;
}

// ---------------------------------------------------------------------------
// Stub implementation
// ---------------------------------------------------------------------------

/// Total simulated VRAM (24 GiB).
const STUB_TOTAL_MEM: usize = 24 * 1024 * 1024 * 1024;

/// Simulated GPU name.
const STUB_GPU_NAME: &str = "OuterLink Virtual GPU";

/// Metadata for a stub CUDA context.
struct StubContext {
    device: i32,
    flags: u32,
}

/// Metadata for a stub loaded module.
struct StubModule {
    data_len: usize,
}

/// Metadata for a stub kernel function.
struct StubFunction {
    module_id: u64,
    name: String,
    /// Settable attributes (e.g. MAX_DYNAMIC_SHARED_SIZE_BYTES).
    attributes: HashMap<i32, i32>,
}

/// Metadata for a stub CUDA stream.
struct StubStream {
    flags: u32,
    priority: i32,
    /// Context that was current when this stream was created.
    ctx: u64,
}

/// Metadata for a stub CUDA event.
struct StubEvent {
    flags: u32,
    recorded: bool,
    timestamp_ns: u64,
}

/// State for a device's primary context.
struct PrimaryCtxState {
    /// Context handle (0 = not created yet).
    ctx_handle: u64,
    /// Reference count.
    refcount: u32,
    /// Flags set via cuDevicePrimaryCtxSetFlags.
    flags: u32,
}

/// Combined state for the stub GPU backend, protected by a single `Mutex`
/// to prevent deadlocks from multi-lock acquisition ordering.
struct StubState {
    /// Simulated VRAM: maps device-pointer -> byte buffer.
    allocations: HashMap<u64, Vec<u8>>,
    /// Monotonically increasing counter for fake device pointers.
    next_ptr: u64,
    /// Simulated CUDA contexts: maps context handle -> context info.
    contexts: HashMap<u64, StubContext>,
    /// Counter for generating context handles (starts at a distinctive address).
    next_ctx_id: u64,
    /// Loaded modules: handle -> module info.
    modules: HashMap<u64, StubModule>,
    /// Kernel functions: handle -> function info.
    functions: HashMap<u64, StubFunction>,
    /// Counter for generating module handles.
    next_module_id: u64,
    /// Counter for generating function handles.
    next_function_id: u64,
    /// CUDA streams: handle -> stream info.
    streams: HashMap<u64, StubStream>,
    /// Counter for generating stream handles.
    next_stream_id: u64,
    /// CUDA events: handle -> event info.
    events: HashMap<u64, StubEvent>,
    /// Counter for generating event handles.
    next_event_id: u64,
    /// Monotonically increasing fake timestamp for events.
    event_timestamp_counter: u64,
    /// Pinned host memory allocations: handle -> byte buffer.
    host_allocations: HashMap<u64, Vec<u8>>,
    /// Counter for generating host memory handles.
    next_host_ptr: u64,
    /// Primary contexts: device ordinal -> primary context state.
    primary_contexts: HashMap<i32, PrimaryCtxState>,
    /// Context stack for push/pop operations.
    context_stack: Vec<u64>,
    /// Resource limits: CU_LIMIT_* -> value.
    context_limits: HashMap<u32, u64>,
    /// Peer contexts currently enabled for P2P access.
    peer_access: HashSet<u64>,
    /// Preferred cache configuration (CU_FUNC_CACHE_PREFER_*).
    cache_config: u32,
    /// Shared memory configuration (CU_SHARED_MEM_CONFIG_*_BANK_SIZE).
    shared_mem_config: u32,
}

impl StubState {
    /// Return how many bytes are currently "allocated".
    fn used_bytes(&self) -> usize {
        self.allocations.values().map(|v| v.len()).sum()
    }
}

/// A fake GPU backend that stores "device memory" in a `HashMap`.
///
/// This is intentionally **not** async -- every method returns instantly.
/// Useful for:
/// * Unit/integration tests that run without hardware.
/// * Validating the protocol and handler logic end-to-end.
pub struct StubGpuBackend {
    /// All mutable state under a single lock to prevent deadlocks.
    state: Mutex<StubState>,
}

impl StubGpuBackend {
    /// Create a new stub backend with no allocations.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(StubState {
                allocations: HashMap::new(),
                // Start at a recognisable fake base address.
                next_ptr: 0x0000_DEAD_0000_0000,
                contexts: HashMap::new(),
                // Start at a distinctive base so context handles are debuggable.
                next_ctx_id: 0xC000_0000_0000_0001,
                modules: HashMap::new(),
                functions: HashMap::new(),
                next_module_id: 0xA000_0000_0000_0001,
                next_function_id: 0xF000_0000_0000_0001,
                streams: HashMap::new(),
                next_stream_id: 0x5000_0000_0000_0001,
                events: HashMap::new(),
                next_event_id: 0xE000_0000_0000_0001,
                event_timestamp_counter: 1000,
                host_allocations: HashMap::new(),
                next_host_ptr: 0x0000_CAFE_0000_0000,
                primary_contexts: HashMap::new(),
                context_stack: Vec::new(),
                context_limits: {
                    let mut m = HashMap::new();
                    m.insert(0x00, 1024);       // CU_LIMIT_STACK_SIZE
                    m.insert(0x01, 1_048_576);   // CU_LIMIT_PRINTF_FIFO_SIZE
                    m.insert(0x02, 8_388_608);   // CU_LIMIT_MALLOC_HEAP_SIZE
                    m
                },
                peer_access: HashSet::new(),
                cache_config: 0,        // CU_FUNC_CACHE_PREFER_NONE
                shared_mem_config: 0,    // CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
            }),
        }
    }

    /// Helper: check that `device` is in the valid range (we expose 1 GPU).
    fn check_device(device: i32) -> Result<(), CuResult> {
        if device == 0 {
            Ok(())
        } else {
            Err(CuResult::InvalidDevice)
        }
    }
}

impl Default for StubGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for StubGpuBackend {
    fn init(&self) -> CuResult {
        CuResult::Success
    }

    fn driver_get_version(&self) -> Result<i32, CuResult> {
        // CUDA 12.4 -> 12040
        Ok(12040)
    }

    fn device_get_count(&self) -> Result<i32, CuResult> {
        Ok(1)
    }

    fn device_get_name(&self, device: i32) -> Result<String, CuResult> {
        Self::check_device(device)?;
        Ok(STUB_GPU_NAME.to_string())
    }

    fn device_get_attribute(&self, attrib: i32, device: i32) -> Result<i32, CuResult> {
        Self::check_device(device)?;
        // Return plausible values that match an RTX 3090-class card.
        let val = match attrib {
            1 => 1024,  // MaxThreadsPerBlock
            2 => 1024,  // MaxBlockDimX
            3 => 1024,  // MaxBlockDimY
            4 => 64,    // MaxBlockDimZ
            5 => 2_147_483_647, // MaxGridDimX
            6 => 65535,  // MaxGridDimY
            7 => 65535,  // MaxGridDimZ
            16 => 82,    // MultiprocessorCount
            75 => 8,     // ComputeCapabilityMajor
            76 => 6,     // ComputeCapabilityMinor
            81 => 102400, // MaxSharedMemoryPerMultiprocessor (100 KB)
            _ => return Err(CuResult::InvalidValue),
        };
        Ok(val)
    }

    fn device_total_mem(&self, device: i32) -> Result<usize, CuResult> {
        Self::check_device(device)?;
        Ok(STUB_TOTAL_MEM)
    }

    fn device_get_uuid(&self, device: i32) -> Result<[u8; 16], CuResult> {
        Self::check_device(device)?;
        // Deterministic fake UUID.
        Ok([
            0x4F, 0x4C, 0x4E, 0x4B, // "OLNK"
            0x00, 0x00, 0x00, 0x00,
            0xDE, 0xAD, 0xBE, 0xEF,
            0x00, 0x00, 0x00, 0x01,
        ])
    }

    fn mem_alloc(&self, size: usize) -> Result<u64, CuResult> {
        if size == 0 {
            return Err(CuResult::InvalidValue);
        }
        // Single lock covers both capacity check and pointer bump, preventing
        // TOCTOU races and eliminating AB/BA deadlock risk.
        let mut state = self.state.lock().unwrap();
        let used = state.used_bytes();
        if used + size > STUB_TOTAL_MEM {
            return Err(CuResult::OutOfMemory);
        }
        let ptr = state.next_ptr;
        state.next_ptr += size as u64;
        state.allocations.insert(ptr, vec![0u8; size]);
        Ok(ptr)
    }

    fn mem_free(&self, ptr: u64) -> Result<(), CuResult> {
        match self.state.lock().unwrap().allocations.remove(&ptr) {
            Some(_) => Ok(()),
            None => Err(CuResult::InvalidValue),
        }
    }

    fn memcpy_htod(&self, dst: u64, data: &[u8]) -> CuResult {
        let mut state = self.state.lock().unwrap();
        match state.allocations.get_mut(&dst) {
            Some(buf) => {
                if data.len() > buf.len() {
                    return CuResult::InvalidValue;
                }
                buf[..data.len()].copy_from_slice(data);
                CuResult::Success
            }
            None => CuResult::InvalidValue,
        }
    }

    fn memcpy_dtoh(&self, src: u64, size: usize) -> Result<Vec<u8>, CuResult> {
        let state = self.state.lock().unwrap();
        match state.allocations.get(&src) {
            Some(buf) => {
                if size > buf.len() {
                    return Err(CuResult::InvalidValue);
                }
                Ok(buf[..size].to_vec())
            }
            None => Err(CuResult::InvalidValue),
        }
    }

    fn mem_get_info(&self) -> Result<(usize, usize), CuResult> {
        let used = self.state.lock().unwrap().used_bytes();
        Ok((STUB_TOTAL_MEM - used, STUB_TOTAL_MEM))
    }

    fn ctx_create(&self, flags: u32, device: i32) -> Result<u64, CuResult> {
        Self::check_device(device)?;
        let mut state = self.state.lock().unwrap();
        let id = state.next_ctx_id;
        state.next_ctx_id += 1;
        state.contexts.insert(id, StubContext { device, flags });
        Ok(id)
    }

    fn ctx_destroy(&self, ctx: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if state.contexts.remove(&ctx).is_none() {
            return Err(CuResult::InvalidContext);
        }
        Ok(())
    }

    fn ctx_exists(&self, ctx: u64) -> bool {
        let state = self.state.lock().unwrap();
        state.contexts.contains_key(&ctx)
    }

    fn ctx_get_device(&self, ctx: u64) -> Result<i32, CuResult> {
        let state = self.state.lock().unwrap();
        match state.contexts.get(&ctx) {
            Some(c) => Ok(c.device),
            None => Err(CuResult::InvalidContext),
        }
    }

    fn ctx_synchronize(&self) -> Result<(), CuResult> {
        // Stub has no asynchronous work to synchronize.
        Ok(())
    }

    // --- Module operations ---

    fn module_load_data(&self, data: &[u8]) -> Result<u64, CuResult> {
        if data.is_empty() {
            return Err(CuResult::InvalidValue);
        }
        let mut state = self.state.lock().unwrap();
        let id = state.next_module_id;
        state.next_module_id += 1;
        state.modules.insert(id, StubModule { data_len: data.len() });
        Ok(id)
    }

    fn module_load_data_ex(&self, data: &[u8], _options: &[(i32, u64)]) -> Result<u64, CuResult> {
        // Stub ignores JIT options — they only matter for real CUDA compilation.
        self.module_load_data(data)
    }

    fn module_unload(&self, module: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if state.modules.remove(&module).is_none() {
            return Err(CuResult::InvalidValue);
        }
        // Remove all functions that belonged to this module.
        state.functions.retain(|_, f| f.module_id != module);
        Ok(())
    }

    fn module_get_function(&self, module: u64, name: &str) -> Result<u64, CuResult> {
        let mut state = self.state.lock().unwrap();
        if !state.modules.contains_key(&module) {
            return Err(CuResult::InvalidValue);
        }
        let id = state.next_function_id;
        state.next_function_id += 1;
        state.functions.insert(id, StubFunction {
            module_id: module,
            name: name.to_string(),
            attributes: HashMap::new(),
        });
        Ok(id)
    }

    fn module_get_global(&self, module: u64, name: &str) -> Result<(u64, usize), CuResult> {
        let mut state = self.state.lock().unwrap();
        if !state.modules.contains_key(&module) {
            return Err(CuResult::InvalidValue);
        }
        let _ = name; // name is accepted but not tracked for globals in the stub
        let ptr = state.next_ptr;
        state.next_ptr += 256;
        state.allocations.insert(ptr, vec![0u8; 256]);
        Ok((ptr, 256))
    }

    fn func_get_attribute(&self, attrib: i32, func: u64) -> Result<i32, CuResult> {
        let state = self.state.lock().unwrap();
        let stub_fn = match state.functions.get(&func) {
            Some(f) => f,
            None => return Err(CuResult::InvalidValue),
        };
        // Check if this attribute was explicitly set via func_set_attribute.
        // Settable attributes: 8 (MAX_DYNAMIC_SHARED_SIZE_BYTES), 9 (PREFERRED_SHARED_MEMORY_CARVEOUT).
        if let Some(&stored) = stub_fn.attributes.get(&attrib) {
            return Ok(stored);
        }
        // Return plausible defaults for a Compute Capability 8.6 kernel.
        let val = match attrib {
            0 => 1024,  // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            1 => 0,     // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES (static)
            2 => 0,     // CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
            3 => 0,     // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
            4 => 32,    // CU_FUNC_ATTRIBUTE_NUM_REGS
            5 => 80,    // CU_FUNC_ATTRIBUTE_PTX_VERSION
            6 => 80,    // CU_FUNC_ATTRIBUTE_BINARY_VERSION
            7 => 0,     // CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
            8 => 0,     // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            9 => 0,     // CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            10 => 0,    // CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_REQUESTED
            11 => 0,    // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
            12 => 0,    // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
            13 => 0,    // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
            14 => 0,    // CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
            15 => 0,    // CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
            _ => return Err(CuResult::InvalidValue),
        };
        Ok(val)
    }

    fn func_set_attribute(&self, func: u64, attrib: i32, value: i32) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        let stub_fn = match state.functions.get_mut(&func) {
            Some(f) => f,
            None => return Err(CuResult::InvalidValue),
        };
        // Only settable attributes: 8 (MAX_DYNAMIC_SHARED_SIZE_BYTES),
        // 9 (PREFERRED_SHARED_MEMORY_CARVEOUT).
        match attrib {
            8 | 9 => {
                stub_fn.attributes.insert(attrib, value);
                Ok(())
            }
            // Valid read-only attributes -- not settable
            0..=7 | 10..=15 => Err(CuResult::InvalidValue),
            _ => Err(CuResult::InvalidValue),
        }
    }

    fn mem_get_address_range(&self, dptr: u64) -> Result<(u64, usize), CuResult> {
        let state = self.state.lock().unwrap();
        // Exact match: dptr is the base of an allocation.
        if let Some(buf) = state.allocations.get(&dptr) {
            return Ok((dptr, buf.len()));
        }
        // In the stub we don't support interior pointer lookup.
        Err(CuResult::InvalidValue)
    }

    // --- Occupancy operations ---

    fn occupancy_max_active_blocks(
        &self,
        func: u64,
        block_size: i32,
        _dynamic_smem_size: u64,
        _flags: u32,
    ) -> Result<i32, CuResult> {
        let state = self.state.lock().unwrap();
        if !state.functions.contains_key(&func) {
            return Err(CuResult::InvalidValue);
        }
        if block_size <= 0 {
            return Err(CuResult::InvalidValue);
        }
        // Simplified Ampere occupancy model
        let blocks = std::cmp::min(STUB_MAX_THREADS_PER_SM / block_size, STUB_MAX_BLOCKS_PER_SM);
        Ok(std::cmp::max(blocks, 1))
    }

    fn occupancy_max_potential_block_size(
        &self,
        func: u64,
        _dynamic_smem_size: u64,
        block_size_limit: i32,
        _flags: u32,
    ) -> Result<(i32, i32), CuResult> {
        let state = self.state.lock().unwrap();
        if !state.functions.contains_key(&func) {
            return Err(CuResult::InvalidValue);
        }
        // Safe default: blockSize=256, numSMs from stub constant (3090)
        let block_size = if block_size_limit > 0 && block_size_limit < 256 {
            block_size_limit
        } else {
            256
        };
        let blocks_per_sm = STUB_MAX_THREADS_PER_SM / block_size;
        let min_grid_size = blocks_per_sm * STUB_NUM_SMS;
        Ok((min_grid_size, block_size))
    }

    // --- Pointer attribute queries ---

    fn pointer_get_attribute(&self, attribute: i32, ptr: u64) -> Result<u64, CuResult> {
        let state = self.state.lock().unwrap();
        let is_device = state.allocations.contains_key(&ptr);
        let is_host = state.host_allocations.contains_key(&ptr);
        if !is_device && !is_host {
            return Err(CuResult::InvalidValue);
        }
        let val = match attribute {
            1 => 0u64,                          // CONTEXT: stub doesn't track
            2 => if is_device { 2 } else { 1 }, // MEMORY_TYPE: DEVICE=2, HOST=1
            3 => if is_device { ptr } else { 0 }, // DEVICE_POINTER
            4 => if is_host { ptr } else { 0 },   // HOST_POINTER
            6 => 0,                              // IS_MANAGED: no managed memory
            8 => 0,                              // DEVICE_ORDINAL: always 0
            _ => return Err(CuResult::InvalidValue),
        };
        Ok(val)
    }

    fn pointer_get_attributes(&self, attributes: &[i32], ptr: u64) -> Result<Vec<u64>, CuResult> {
        // Validate the pointer exists before iterating
        {
            let state = self.state.lock().unwrap();
            let is_device = state.allocations.contains_key(&ptr);
            let is_host = state.host_allocations.contains_key(&ptr);
            if !is_device && !is_host {
                return Err(CuResult::InvalidValue);
            }
        }
        let mut results = Vec::with_capacity(attributes.len());
        for &attr in attributes {
            results.push(self.pointer_get_attribute(attr, ptr)?);
        }
        Ok(results)
    }

    // --- Stream operations ---

    fn stream_create(&self, flags: u32) -> Result<u64, CuResult> {
        let mut state = self.state.lock().unwrap();
        let id = state.next_stream_id;
        state.next_stream_id += 1;
        state.streams.insert(id, StubStream { flags, priority: 0, ctx: 0 });
        Ok(id)
    }

    fn stream_destroy(&self, stream: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if state.streams.remove(&stream).is_none() {
            return Err(CuResult::InvalidValue);
        }
        Ok(())
    }

    fn stream_synchronize(&self, stream: u64) -> Result<(), CuResult> {
        let state = self.state.lock().unwrap();
        if !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        Ok(())
    }

    fn stream_query(&self, stream: u64) -> Result<(), CuResult> {
        let state = self.state.lock().unwrap();
        if !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        // Stub: all work is always complete.
        Ok(())
    }

    fn stream_create_with_priority(&self, flags: u32, priority: i32, ctx: u64) -> Result<u64, CuResult> {
        let mut state = self.state.lock().unwrap();
        let id = state.next_stream_id;
        state.next_stream_id += 1;
        state.streams.insert(id, StubStream { flags, priority, ctx });
        Ok(id)
    }

    fn stream_get_priority(&self, stream: u64) -> Result<i32, CuResult> {
        let state = self.state.lock().unwrap();
        match state.streams.get(&stream) {
            Some(s) => Ok(s.priority),
            None => Err(CuResult::InvalidValue),
        }
    }

    fn stream_get_flags(&self, stream: u64) -> Result<u32, CuResult> {
        let state = self.state.lock().unwrap();
        match state.streams.get(&stream) {
            Some(s) => Ok(s.flags),
            None => Err(CuResult::InvalidValue),
        }
    }

    fn stream_get_ctx(&self, stream: u64) -> Result<u64, CuResult> {
        let state = self.state.lock().unwrap();
        match state.streams.get(&stream) {
            Some(s) => Ok(s.ctx),
            None => Err(CuResult::InvalidValue),
        }
    }

    // --- Event operations ---

    fn event_create(&self, flags: u32) -> Result<u64, CuResult> {
        let mut state = self.state.lock().unwrap();
        let id = state.next_event_id;
        state.next_event_id += 1;
        state.events.insert(id, StubEvent {
            flags,
            recorded: false,
            timestamp_ns: 0,
        });
        Ok(id)
    }

    fn event_destroy(&self, event: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if state.events.remove(&event).is_none() {
            return Err(CuResult::InvalidValue);
        }
        Ok(())
    }

    fn event_record(&self, event: u64, stream: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        // Validate stream exists (0 = default/null stream is always valid).
        if stream != 0 && !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        let ts = state.event_timestamp_counter;
        state.event_timestamp_counter += 100_000; // 0.1ms in nanoseconds
        match state.events.get_mut(&event) {
            Some(ev) => {
                ev.recorded = true;
                ev.timestamp_ns = ts;
                Ok(())
            }
            None => Err(CuResult::InvalidValue),
        }
    }

    fn event_synchronize(&self, event: u64) -> Result<(), CuResult> {
        let state = self.state.lock().unwrap();
        if !state.events.contains_key(&event) {
            return Err(CuResult::InvalidValue);
        }
        // Stub: always complete immediately.
        Ok(())
    }

    fn event_elapsed_time(&self, start: u64, end: u64) -> Result<f32, CuResult> {
        let state = self.state.lock().unwrap();
        let start_ev = state.events.get(&start).ok_or(CuResult::InvalidValue)?;
        let end_ev = state.events.get(&end).ok_or(CuResult::InvalidValue)?;
        if !start_ev.recorded || !end_ev.recorded {
            return Err(CuResult::NotReady);
        }
        // Real CUDA returns negative elapsed time when end < start (no error).
        let diff_ns = end_ev.timestamp_ns as f64 - start_ev.timestamp_ns as f64;
        Ok((diff_ns / 1_000_000.0) as f32)
    }

    fn event_query(&self, event: u64) -> Result<(), CuResult> {
        let state = self.state.lock().unwrap();
        if !state.events.contains_key(&event) {
            return Err(CuResult::InvalidValue);
        }
        // Stub: events are always "complete".
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64, _flags: u32) -> Result<(), CuResult> {
        let state = self.state.lock().unwrap();
        // stream 0 = default stream, always valid.
        if stream != 0 && !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        if !state.events.contains_key(&event) {
            return Err(CuResult::InvalidValue);
        }
        // Stub: no actual waiting.
        Ok(())
    }

    // --- Memory: host pinned ---

    fn mem_alloc_host(&self, size: usize) -> Result<u64, CuResult> {
        if size == 0 {
            return Err(CuResult::InvalidValue);
        }
        let mut state = self.state.lock().unwrap();
        let ptr = state.next_host_ptr;
        state.next_host_ptr += size as u64;
        state.host_allocations.insert(ptr, vec![0u8; size]);
        Ok(ptr)
    }

    fn mem_free_host(&self, ptr: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if state.host_allocations.remove(&ptr).is_none() {
            return Err(CuResult::InvalidValue);
        }
        Ok(())
    }

    // --- Memory: async copy ---

    fn memcpy_htod_async(&self, dst: u64, data: &[u8], stream: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        // Validate stream (0 = default stream, always valid).
        if stream != 0 && !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        // Stub: async behaves same as sync (no real streams).
        match state.allocations.get_mut(&dst) {
            Some(buf) => {
                if data.len() > buf.len() {
                    return Err(CuResult::InvalidValue);
                }
                buf[..data.len()].copy_from_slice(data);
                Ok(())
            }
            None => Err(CuResult::InvalidValue),
        }
    }

    fn memcpy_dtoh_async(&self, src: u64, size: usize, stream: u64) -> Result<Vec<u8>, CuResult> {
        let state = self.state.lock().unwrap();
        // Validate stream (0 = default stream, always valid).
        if stream != 0 && !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        // Stub: async behaves same as sync (no real streams).
        match state.allocations.get(&src) {
            Some(buf) => {
                if size > buf.len() {
                    return Err(CuResult::InvalidValue);
                }
                Ok(buf[..size].to_vec())
            }
            None => Err(CuResult::InvalidValue),
        }
    }

    // --- Memory: memset ---

    fn memset_d8(&self, dst: u64, value: u8, count: usize) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        match state.allocations.get_mut(&dst) {
            Some(buf) => {
                if count > buf.len() {
                    return Err(CuResult::InvalidValue);
                }
                for b in &mut buf[..count] {
                    *b = value;
                }
                Ok(())
            }
            None => Err(CuResult::InvalidValue),
        }
    }

    fn memset_d32(&self, dst: u64, value: u32, count: usize) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        match state.allocations.get_mut(&dst) {
            Some(buf) => {
                let byte_count = count * 4;
                if byte_count > buf.len() {
                    return Err(CuResult::InvalidValue);
                }
                let val_bytes = value.to_le_bytes();
                for i in 0..count {
                    let offset = i * 4;
                    buf[offset..offset + 4].copy_from_slice(&val_bytes);
                }
                Ok(())
            }
            None => Err(CuResult::InvalidValue),
        }
    }

    fn memset_d8_async(&self, dst: u64, value: u8, count: usize, stream: u64) -> Result<(), CuResult> {
        {
            let state = self.state.lock().unwrap();
            // Validate stream (0 = default stream, always valid).
            if stream != 0 && !state.streams.contains_key(&stream) {
                return Err(CuResult::InvalidValue);
            }
        }
        // Stub: async behaves same as sync.
        self.memset_d8(dst, value, count)
    }

    fn memset_d32_async(&self, dst: u64, value: u32, count: usize, stream: u64) -> Result<(), CuResult> {
        {
            let state = self.state.lock().unwrap();
            // Validate stream (0 = default stream, always valid).
            if stream != 0 && !state.streams.contains_key(&stream) {
                return Err(CuResult::InvalidValue);
            }
        }
        // Stub: async behaves same as sync.
        self.memset_d32(dst, value, count)
    }

    // --- Memory: device-to-device ---

    fn memcpy_dtod(&self, dst: u64, src: u64, size: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        // Read source data first.
        let src_buf = state.allocations.get(&src).ok_or(CuResult::InvalidValue)?;
        if (size as usize) > src_buf.len() {
            return Err(CuResult::InvalidValue);
        }
        let data = src_buf[..size as usize].to_vec();
        // Write to destination.
        let dst_buf = state.allocations.get_mut(&dst).ok_or(CuResult::InvalidValue)?;
        if (size as usize) > dst_buf.len() {
            return Err(CuResult::InvalidValue);
        }
        dst_buf[..size as usize].copy_from_slice(&data);
        Ok(())
    }

    // --- Kernel launch ---

    fn launch_kernel(
        &self,
        func: u64,
        _grid_dim: [u32; 3],
        _block_dim: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        _params: &[u8],
    ) -> Result<(), CuResult> {
        let state = self.state.lock().unwrap();
        // Validate function exists.
        if !state.functions.contains_key(&func) {
            return Err(CuResult::InvalidValue);
        }
        // Validate stream (0 = default stream is always valid).
        if stream != 0 && !state.streams.contains_key(&stream) {
            return Err(CuResult::InvalidValue);
        }
        tracing::trace!(func, stream, "stub: launch_kernel (no-op)");
        Ok(())
    }

    // --- Peer access ---

    fn device_can_access_peer(&self, dev: i32, peer_dev: i32) -> Result<i32, CuResult> {
        Self::check_device(dev)?;
        Self::check_device(peer_dev)?;
        // Per CUDA spec: a device cannot peer-access itself (that's just normal access).
        // Return 0 if dev == peer_dev.
        // In stub single-device mode, dev and peer_dev are both 0 so we always return 0.
        if dev == peer_dev {
            Ok(0)
        } else {
            // Different devices on the same server: would be 1 if we had multiple GPUs.
            // Single-device stub: can never reach here because check_device(dev) only allows 0.
            Ok(0)
        }
    }

    fn device_get_p2p_attribute(&self, _attrib: i32, src_device: i32, dst_device: i32) -> Result<i32, CuResult> {
        Self::check_device(src_device)?;
        Self::check_device(dst_device)?;
        // Stub: return 0 for all attributes (no P2P support).
        Ok(0)
    }

    fn ctx_enable_peer_access(&self, peer_ctx: u64, _flags: u32) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if !state.contexts.contains_key(&peer_ctx) {
            return Err(CuResult::InvalidContext);
        }
        if state.peer_access.contains(&peer_ctx) {
            return Err(CuResult::PeerAccessAlreadyEnabled);
        }
        state.peer_access.insert(peer_ctx);
        Ok(())
    }

    fn ctx_disable_peer_access(&self, peer_ctx: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if !state.peer_access.remove(&peer_ctx) {
            return Err(CuResult::PeerAccessNotEnabled);
        }
        Ok(())
    }

    // --- Context stack & query operations ---

    fn ctx_push_current(&self, ctx: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if !state.contexts.contains_key(&ctx) {
            return Err(CuResult::InvalidContext);
        }
        state.context_stack.push(ctx);
        Ok(())
    }

    fn ctx_pop_current(&self) -> Result<u64, CuResult> {
        let mut state = self.state.lock().unwrap();
        state.context_stack.pop().ok_or(CuResult::InvalidContext)
    }

    fn ctx_get_api_version(&self, ctx: u64) -> Result<u32, CuResult> {
        let state = self.state.lock().unwrap();
        if !state.contexts.contains_key(&ctx) {
            return Err(CuResult::InvalidContext);
        }
        // Simulated CUDA 12.0 API version
        Ok(12000)
    }

    fn ctx_get_limit(&self, limit: u32) -> Result<u64, CuResult> {
        let state = self.state.lock().unwrap();
        state.context_limits.get(&limit).copied().ok_or(CuResult::InvalidValue)
    }

    fn ctx_set_limit(&self, limit: u32, value: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        state.context_limits.insert(limit, value);
        Ok(())
    }

    fn ctx_get_stream_priority_range(&self) -> Result<(i32, i32), CuResult> {
        // Standard NVIDIA range: 0 = lowest priority, -1 = highest priority
        Ok((0, -1))
    }

    fn ctx_get_flags(&self, ctx: u64) -> Result<u32, CuResult> {
        let state = self.state.lock().unwrap();
        match state.contexts.get(&ctx) {
            Some(c) => Ok(c.flags),
            None => Err(CuResult::InvalidContext),
        }
    }

    fn ctx_get_cache_config(&self) -> Result<u32, CuResult> {
        let state = self.state.lock().unwrap();
        Ok(state.cache_config)
    }

    fn ctx_set_cache_config(&self, config: u32) -> Result<(), CuResult> {
        if config > 0x03 {
            return Err(CuResult::InvalidValue);
        }
        let mut state = self.state.lock().unwrap();
        state.cache_config = config;
        Ok(())
    }

    fn ctx_get_shared_mem_config(&self) -> Result<u32, CuResult> {
        let state = self.state.lock().unwrap();
        Ok(state.shared_mem_config)
    }

    fn ctx_set_shared_mem_config(&self, config: u32) -> Result<(), CuResult> {
        if config > 0x02 {
            return Err(CuResult::InvalidValue);
        }
        let mut state = self.state.lock().unwrap();
        state.shared_mem_config = config;
        Ok(())
    }

    fn func_set_cache_config(&self, func: u64, config: u32) -> Result<(), CuResult> {
        if config > 0x03 {
            return Err(CuResult::InvalidValue);
        }
        let mut state = self.state.lock().unwrap();
        let stub_fn = match state.functions.get_mut(&func) {
            Some(f) => f,
            None => return Err(CuResult::InvalidValue),
        };
        // Store as a negative attribute key to avoid collision with CUfunction_attribute
        stub_fn.attributes.insert(-1, config as i32); // -1 = cache config
        Ok(())
    }

    fn func_set_shared_mem_config(&self, func: u64, config: u32) -> Result<(), CuResult> {
        if config > 0x02 {
            return Err(CuResult::InvalidValue);
        }
        let mut state = self.state.lock().unwrap();
        let stub_fn = match state.functions.get_mut(&func) {
            Some(f) => f,
            None => return Err(CuResult::InvalidValue),
        };
        stub_fn.attributes.insert(-2, config as i32); // -2 = shared mem config
        Ok(())
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().unwrap();
        tracing::info!(
            contexts = state.contexts.len(),
            allocations = state.allocations.len(),
            modules = state.modules.len(),
            streams = state.streams.len(),
            events = state.events.len(),
            host_allocations = state.host_allocations.len(),
            "stub: shutdown — releasing all resources"
        );
        state.contexts.clear();
        state.allocations.clear();
        state.modules.clear();
        state.functions.clear();
        state.streams.clear();
        state.events.clear();
        state.host_allocations.clear();
        state.primary_contexts.clear();
        state.context_stack.clear();
        state.context_limits.clear();
    }

    fn primary_ctx_retain(&self, device: i32) -> Result<u64, CuResult> {
        Self::check_device(device)?;
        let mut state = self.state.lock().unwrap();
        // Ensure entry exists.
        if !state.primary_contexts.contains_key(&device) {
            state.primary_contexts.insert(device, PrimaryCtxState {
                ctx_handle: 0,
                refcount: 0,
                flags: 0,
            });
        }
        let needs_create = state.primary_contexts[&device].ctx_handle == 0;
        if needs_create {
            let id = state.next_ctx_id;
            state.next_ctx_id += 1;
            let flags = state.primary_contexts[&device].flags;
            state.contexts.insert(id, StubContext { device, flags });
            state.primary_contexts.get_mut(&device).unwrap().ctx_handle = id;
        }
        let entry = state.primary_contexts.get_mut(&device).unwrap();
        entry.refcount += 1;
        Ok(entry.ctx_handle)
    }

    fn primary_ctx_release(&self, device: i32) -> Result<(), CuResult> {
        Self::check_device(device)?;
        let mut state = self.state.lock().unwrap();
        let (refcount, ctx_handle) = match state.primary_contexts.get(&device) {
            Some(e) if e.refcount > 0 => (e.refcount, e.ctx_handle),
            _ => return Err(CuResult::InvalidContext),
        };
        let new_refcount = refcount - 1;
        if new_refcount == 0 {
            state.contexts.remove(&ctx_handle);
            let entry = state.primary_contexts.get_mut(&device).unwrap();
            entry.refcount = 0;
            entry.ctx_handle = 0;
        } else {
            state.primary_contexts.get_mut(&device).unwrap().refcount = new_refcount;
        }
        Ok(())
    }

    fn primary_ctx_get_state(&self, device: i32) -> Result<(u32, i32), CuResult> {
        Self::check_device(device)?;
        let state = self.state.lock().unwrap();
        match state.primary_contexts.get(&device) {
            Some(entry) => {
                let active = if entry.refcount > 0 { 1 } else { 0 };
                Ok((entry.flags, active))
            }
            None => Ok((0, 0)),
        }
    }

    fn primary_ctx_set_flags(&self, device: i32, flags: u32) -> Result<(), CuResult> {
        Self::check_device(device)?;
        let mut state = self.state.lock().unwrap();
        if !state.primary_contexts.contains_key(&device) {
            state.primary_contexts.insert(device, PrimaryCtxState {
                ctx_handle: 0,
                refcount: 0,
                flags: 0,
            });
        }
        let entry = state.primary_contexts.get_mut(&device).unwrap();
        if entry.refcount > 0 {
            return Err(CuResult::PrimaryContextActive);
        }
        entry.flags = flags;
        Ok(())
    }

    fn primary_ctx_reset(&self, device: i32) -> Result<Option<u64>, CuResult> {
        Self::check_device(device)?;
        let mut state = self.state.lock().unwrap();
        let (old_handle, had_ctx) = match state.primary_contexts.get(&device) {
            Some(entry) if entry.ctx_handle != 0 => (entry.ctx_handle, true),
            Some(_) => (0, false),
            None => return Ok(None),
        };
        if had_ctx {
            state.contexts.remove(&old_handle);
        }
        let entry = state.primary_contexts.get_mut(&device).unwrap();
        entry.ctx_handle = 0;
        entry.refcount = 0;
        // Preserve flags
        Ok(if had_ctx { Some(old_handle) } else { None })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_succeeds() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.init(), CuResult::Success);
    }

    #[test]
    fn test_driver_version() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.driver_get_version().unwrap(), 12040);
    }

    #[test]
    fn test_device_count() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.device_get_count().unwrap(), 1);
    }

    #[test]
    fn test_device_name() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.device_get_name(0).unwrap(), "OuterLink Virtual GPU");
        assert_eq!(gpu.device_get_name(1), Err(CuResult::InvalidDevice));
    }

    #[test]
    fn test_device_attribute_compute_cap() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.device_get_attribute(75, 0).unwrap(), 8); // major
        assert_eq!(gpu.device_get_attribute(76, 0).unwrap(), 6); // minor
    }

    #[test]
    fn test_device_attribute_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.device_get_attribute(99999, 0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_device_total_mem() {
        let gpu = StubGpuBackend::new();
        let mem = gpu.device_total_mem(0).unwrap();
        assert_eq!(mem, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_device_uuid() {
        let gpu = StubGpuBackend::new();
        let uuid = gpu.device_get_uuid(0).unwrap();
        assert_eq!(&uuid[..4], b"OLNK");
    }

    #[test]
    fn test_mem_alloc_and_free() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(1024).unwrap();
        assert_ne!(ptr, 0);
        assert!(gpu.mem_free(ptr).is_ok());
        // Double free should fail.
        assert_eq!(gpu.mem_free(ptr), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_mem_alloc_zero_size() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.mem_alloc(0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_roundtrip() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(256).unwrap();

        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        assert_eq!(gpu.memcpy_htod(ptr, &data), CuResult::Success);

        let out = gpu.memcpy_dtoh(ptr, 256).unwrap();
        assert_eq!(out, data);

        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_memcpy_htod_overflow() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        let big = vec![0u8; 32];
        assert_eq!(gpu.memcpy_htod(ptr, &big), CuResult::InvalidValue);
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_memcpy_dtoh_overflow() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        assert_eq!(gpu.memcpy_dtoh(ptr, 32), Err(CuResult::InvalidValue));
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_mem_get_info() {
        let gpu = StubGpuBackend::new();
        let (free, total) = gpu.mem_get_info().unwrap();
        assert_eq!(total, 24 * 1024 * 1024 * 1024);
        assert_eq!(free, total);

        let ptr = gpu.mem_alloc(1024).unwrap();
        let (free2, total2) = gpu.mem_get_info().unwrap();
        assert_eq!(total2, total);
        assert_eq!(free2, total - 1024);

        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_ctx_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert_ne!(ctx, 0);
        // Context should exist in the backend.
        assert!(gpu.ctx_exists(ctx));
        // Destroy it.
        assert!(gpu.ctx_destroy(ctx).is_ok());
        // Context should no longer exist.
        assert!(!gpu.ctx_exists(ctx));
    }

    #[test]
    fn test_ctx_exists() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert!(gpu.ctx_exists(ctx));
        // Non-existent handle should return false.
        assert!(!gpu.ctx_exists(0xBAAD));
        // After destroy, should return false.
        gpu.ctx_destroy(ctx).unwrap();
        assert!(!gpu.ctx_exists(ctx));
    }

    #[test]
    fn test_ctx_get_device() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert_eq!(gpu.ctx_get_device(ctx).unwrap(), 0);
        // Invalid context should error.
        assert_eq!(gpu.ctx_get_device(0xBAAD), Err(CuResult::InvalidContext));
    }

    #[test]
    fn test_ctx_double_destroy() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert!(gpu.ctx_destroy(ctx).is_ok());
        // Second destroy should fail.
        assert_eq!(gpu.ctx_destroy(ctx), Err(CuResult::InvalidContext));
    }

    // ----- Module operation tests -----

    #[test]
    fn test_module_load_and_unload() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"fake ptx data").unwrap();
        assert_ne!(module, 0);
        assert!(gpu.module_unload(module).is_ok());
        // Double unload should fail.
        assert_eq!(gpu.module_unload(module), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_module_load_empty_data() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.module_load_data(b""), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_module_load_data_ex_empty_options() {
        let gpu = StubGpuBackend::new();
        let handle = gpu.module_load_data_ex(b"fake ptx", &[]).unwrap();
        assert_ne!(handle, 0);
        // Module should be usable for get_function just like a normal load
        let func = gpu.module_get_function(handle, "kern").unwrap();
        assert_ne!(func, 0);
    }

    #[test]
    fn test_module_load_data_ex_with_options() {
        let gpu = StubGpuBackend::new();
        // Options are ignored in stub mode but should not cause errors
        let options = vec![
            (0, 32),   // CU_JIT_MAX_REGISTERS = 32
            (7, 4),    // CU_JIT_OPTIMIZATION_LEVEL = 4
        ];
        let handle = gpu.module_load_data_ex(b"ptx code", &options).unwrap();
        assert_ne!(handle, 0);
    }

    #[test]
    fn test_module_load_data_ex_empty_data() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.module_load_data_ex(b"", &[]), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_module_get_function() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"fake ptx").unwrap();
        let func = gpu.module_get_function(module, "my_kernel").unwrap();
        assert_ne!(func, 0);
    }

    #[test]
    fn test_module_get_function_invalid_module() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.module_get_function(0xBAD, "kern"), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_module_get_global() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"fake ptx").unwrap();
        let (dptr, size) = gpu.module_get_global(module, "my_global").unwrap();
        assert_ne!(dptr, 0);
        assert_eq!(size, 256);
    }

    #[test]
    fn test_module_get_global_tracks_allocation() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"fake ptx").unwrap();
        let (dptr, size) = gpu.module_get_global(module, "my_global").unwrap();
        assert_eq!(size, 256);
        // The returned pointer must be usable with memcpy operations.
        let data = vec![42u8; 256];
        assert_eq!(gpu.memcpy_htod(dptr, &data), CuResult::Success);
        let out = gpu.memcpy_dtoh(dptr, 256).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_module_get_global_invalid_module() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.module_get_global(0xBAD, "g"), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_module_unload_removes_functions() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"fake ptx").unwrap();
        let func = gpu.module_get_function(module, "kern").unwrap();
        // Function should be usable for launch_kernel validation.
        // After unloading the module, the function should be gone.
        gpu.module_unload(module).unwrap();
        // launch_kernel with the old function handle should fail.
        assert_eq!(
            gpu.launch_kernel(func, [1,1,1], [1,1,1], 0, 0, &[]),
            Err(CuResult::InvalidValue)
        );
    }

    // ----- Stream operation tests -----

    #[test]
    fn test_stream_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create(0).unwrap();
        assert_ne!(stream, 0);
        assert!(gpu.stream_destroy(stream).is_ok());
        // Double destroy should fail.
        assert_eq!(gpu.stream_destroy(stream), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_stream_synchronize() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create(0).unwrap();
        assert!(gpu.stream_synchronize(stream).is_ok());
    }

    #[test]
    fn test_stream_synchronize_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.stream_synchronize(0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_stream_query() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create(0).unwrap();
        assert!(gpu.stream_query(stream).is_ok());
    }

    #[test]
    fn test_stream_query_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.stream_query(0xBAD), Err(CuResult::InvalidValue));
    }

    // ----- Event operation tests -----

    #[test]
    fn test_event_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        assert_ne!(event, 0);
        assert!(gpu.event_destroy(event).is_ok());
        // Double destroy should fail.
        assert_eq!(gpu.event_destroy(event), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_event_record() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        let stream = gpu.stream_create(0).unwrap();
        assert!(gpu.event_record(event, stream).is_ok());
    }

    #[test]
    fn test_event_record_default_stream() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        // stream=0 is the default stream, always valid.
        assert!(gpu.event_record(event, 0).is_ok());
    }

    #[test]
    fn test_event_record_invalid_event() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.event_record(0xBAD, 0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_event_record_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        assert_eq!(gpu.event_record(event, 0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_event_synchronize() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        assert!(gpu.event_synchronize(event).is_ok());
    }

    #[test]
    fn test_event_synchronize_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.event_synchronize(0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_event_elapsed_time() {
        let gpu = StubGpuBackend::new();
        let e1 = gpu.event_create(0).unwrap();
        let e2 = gpu.event_create(0).unwrap();
        gpu.event_record(e1, 0).unwrap();
        gpu.event_record(e2, 0).unwrap();
        let ms = gpu.event_elapsed_time(e1, e2).unwrap();
        assert!(ms > 0.0, "elapsed time should be positive");
    }

    #[test]
    fn test_event_elapsed_time_negative() {
        let gpu = StubGpuBackend::new();
        let e1 = gpu.event_create(0).unwrap();
        let e2 = gpu.event_create(0).unwrap();
        // Record e1 first, then e2 (e2 has a later timestamp).
        gpu.event_record(e1, 0).unwrap();
        gpu.event_record(e2, 0).unwrap();
        // Passing them in reverse order (start=e2, end=e1) produces negative time.
        // Real CUDA returns a negative elapsed time, not an error.
        let ms = gpu.event_elapsed_time(e2, e1).unwrap();
        assert!(ms < 0.0, "reversed event order should produce negative elapsed time");
    }

    #[test]
    fn test_event_elapsed_time_not_recorded() {
        let gpu = StubGpuBackend::new();
        let e1 = gpu.event_create(0).unwrap();
        let e2 = gpu.event_create(0).unwrap();
        // Neither recorded yet — CUDA spec says NotReady.
        assert_eq!(gpu.event_elapsed_time(e1, e2), Err(CuResult::NotReady));
    }

    #[test]
    fn test_event_query() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        assert!(gpu.event_query(event).is_ok());
    }

    #[test]
    fn test_event_query_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.event_query(0xBAD), Err(CuResult::InvalidValue));
    }

    // ----- LaunchKernel tests -----

    #[test]
    fn test_launch_kernel() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "kern").unwrap();
        assert!(gpu.launch_kernel(func, [1,1,1], [256,1,1], 0, 0, &[]).is_ok());
    }

    #[test]
    fn test_launch_kernel_with_stream() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "kern").unwrap();
        let stream = gpu.stream_create(0).unwrap();
        assert!(gpu.launch_kernel(func, [1,1,1], [256,1,1], 0, stream, &[]).is_ok());
    }

    #[test]
    fn test_launch_kernel_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(
            gpu.launch_kernel(0xBAD, [1,1,1], [1,1,1], 0, 0, &[]),
            Err(CuResult::InvalidValue)
        );
    }

    #[test]
    fn test_launch_kernel_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "kern").unwrap();
        assert_eq!(
            gpu.launch_kernel(func, [1,1,1], [1,1,1], 0, 0xBAD, &[]),
            Err(CuResult::InvalidValue)
        );
    }

    // ----- MemcpyDtoD tests -----

    #[test]
    fn test_memcpy_dtod() {
        let gpu = StubGpuBackend::new();
        let src = gpu.mem_alloc(64).unwrap();
        let dst = gpu.mem_alloc(64).unwrap();

        // Write pattern to source.
        let data: Vec<u8> = (0..64).collect();
        assert_eq!(gpu.memcpy_htod(src, &data), CuResult::Success);

        // Copy device-to-device.
        assert!(gpu.memcpy_dtod(dst, src, 64).is_ok());

        // Read back from destination.
        let out = gpu.memcpy_dtoh(dst, 64).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_memcpy_dtod_partial() {
        let gpu = StubGpuBackend::new();
        let src = gpu.mem_alloc(64).unwrap();
        let dst = gpu.mem_alloc(64).unwrap();

        let data: Vec<u8> = (0..64).collect();
        gpu.memcpy_htod(src, &data);

        // Copy only 32 bytes.
        assert!(gpu.memcpy_dtod(dst, src, 32).is_ok());

        let out = gpu.memcpy_dtoh(dst, 64).unwrap();
        assert_eq!(&out[..32], &data[..32]);
        // Rest should be zeros.
        assert_eq!(&out[32..], &[0u8; 32]);
    }

    #[test]
    fn test_memcpy_dtod_invalid_src() {
        let gpu = StubGpuBackend::new();
        let dst = gpu.mem_alloc(64).unwrap();
        assert_eq!(gpu.memcpy_dtod(dst, 0xBAD, 64), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_dtod_invalid_dst() {
        let gpu = StubGpuBackend::new();
        let src = gpu.mem_alloc(64).unwrap();
        assert_eq!(gpu.memcpy_dtod(0xBAD, src, 64), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_dtod_size_exceeds_src() {
        let gpu = StubGpuBackend::new();
        let src = gpu.mem_alloc(32).unwrap();
        let dst = gpu.mem_alloc(64).unwrap();
        assert_eq!(gpu.memcpy_dtod(dst, src, 64), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_dtod_size_exceeds_dst() {
        let gpu = StubGpuBackend::new();
        let src = gpu.mem_alloc(64).unwrap();
        let dst = gpu.mem_alloc(32).unwrap();
        assert_eq!(gpu.memcpy_dtod(dst, src, 64), Err(CuResult::InvalidValue));
    }

    // ----- MemAllocHost / MemFreeHost tests -----

    #[test]
    fn test_mem_alloc_host() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc_host(4096).unwrap();
        assert_ne!(ptr, 0);
    }

    #[test]
    fn test_mem_alloc_host_zero_size() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.mem_alloc_host(0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_mem_free_host() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc_host(4096).unwrap();
        assert!(gpu.mem_free_host(ptr).is_ok());
    }

    #[test]
    fn test_mem_free_host_double_free() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc_host(4096).unwrap();
        assert!(gpu.mem_free_host(ptr).is_ok());
        assert_eq!(gpu.mem_free_host(ptr), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_mem_free_host_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.mem_free_host(0xBAD), Err(CuResult::InvalidValue));
    }

    // ----- StreamWaitEvent tests -----

    #[test]
    fn test_stream_wait_event() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create(0).unwrap();
        let event = gpu.event_create(0).unwrap();
        assert!(gpu.stream_wait_event(stream, event, 0).is_ok());
    }

    #[test]
    fn test_stream_wait_event_default_stream() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        // stream=0 is the default stream, always valid.
        assert!(gpu.stream_wait_event(0, event, 0).is_ok());
    }

    #[test]
    fn test_stream_wait_event_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let event = gpu.event_create(0).unwrap();
        assert_eq!(gpu.stream_wait_event(0xBAD, event, 0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_stream_wait_event_invalid_event() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create(0).unwrap();
        assert_eq!(gpu.stream_wait_event(stream, 0xBAD, 0), Err(CuResult::InvalidValue));
    }

    // ----- Async memcpy tests -----

    #[test]
    fn test_memcpy_htod_async() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(64).unwrap();
        let stream = gpu.stream_create(0).unwrap();
        let data: Vec<u8> = (0..64).collect();
        assert!(gpu.memcpy_htod_async(ptr, &data, stream).is_ok());
        // Verify the data was written.
        let out = gpu.memcpy_dtoh(ptr, 64).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_memcpy_htod_async_default_stream() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(32).unwrap();
        let data = vec![0xABu8; 32];
        // stream=0 is the default stream, always valid.
        assert!(gpu.memcpy_htod_async(ptr, &data, 0).is_ok());
    }

    #[test]
    fn test_memcpy_htod_async_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(32).unwrap();
        let data = vec![0xABu8; 32];
        assert_eq!(gpu.memcpy_htod_async(ptr, &data, 0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_htod_async_invalid_dst() {
        let gpu = StubGpuBackend::new();
        let data = vec![0xABu8; 32];
        assert_eq!(gpu.memcpy_htod_async(0xBAD, &data, 0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_dtoh_async() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(64).unwrap();
        let stream = gpu.stream_create(0).unwrap();
        let data: Vec<u8> = (0..64).collect();
        gpu.memcpy_htod(ptr, &data);
        let out = gpu.memcpy_dtoh_async(ptr, 64, stream).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_memcpy_dtoh_async_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(32).unwrap();
        assert_eq!(gpu.memcpy_dtoh_async(ptr, 32, 0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memcpy_dtoh_async_invalid_src() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.memcpy_dtoh_async(0xBAD, 32, 0), Err(CuResult::InvalidValue));
    }

    // ----- Memset tests -----

    #[test]
    fn test_memset_d8() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(64).unwrap();
        assert!(gpu.memset_d8(ptr, 0xAB, 64).is_ok());
        let out = gpu.memcpy_dtoh(ptr, 64).unwrap();
        assert!(out.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_memset_d8_partial() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(64).unwrap();
        assert!(gpu.memset_d8(ptr, 0xCC, 32).is_ok());
        let out = gpu.memcpy_dtoh(ptr, 64).unwrap();
        assert!(out[..32].iter().all(|&b| b == 0xCC));
        assert!(out[32..].iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_memset_d8_overflow() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        assert_eq!(gpu.memset_d8(ptr, 0xFF, 32), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memset_d8_invalid_ptr() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.memset_d8(0xBAD, 0xFF, 16), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memset_d32() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        assert!(gpu.memset_d32(ptr, 0xDEADBEEF, 4).is_ok());
        let out = gpu.memcpy_dtoh(ptr, 16).unwrap();
        let val_bytes = 0xDEADBEEFu32.to_le_bytes();
        for i in 0..4 {
            assert_eq!(&out[i*4..i*4+4], &val_bytes);
        }
    }

    #[test]
    fn test_memset_d32_overflow() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(8).unwrap();
        // 3 u32 elements = 12 bytes, but only 8 allocated.
        assert_eq!(gpu.memset_d32(ptr, 0xFF, 3), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memset_d32_invalid_ptr() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.memset_d32(0xBAD, 0xFF, 4), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memset_d8_async() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(64).unwrap();
        let stream = gpu.stream_create(0).unwrap();
        assert!(gpu.memset_d8_async(ptr, 0xDD, 64, stream).is_ok());
        let out = gpu.memcpy_dtoh(ptr, 64).unwrap();
        assert!(out.iter().all(|&b| b == 0xDD));
    }

    #[test]
    fn test_memset_d8_async_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(64).unwrap();
        assert_eq!(gpu.memset_d8_async(ptr, 0xDD, 64, 0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_memset_d32_async() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        let stream = gpu.stream_create(0).unwrap();
        assert!(gpu.memset_d32_async(ptr, 0xCAFEBABE, 4, stream).is_ok());
        let out = gpu.memcpy_dtoh(ptr, 16).unwrap();
        let val_bytes = 0xCAFEBABEu32.to_le_bytes();
        for i in 0..4 {
            assert_eq!(&out[i*4..i*4+4], &val_bytes);
        }
    }

    #[test]
    fn test_memset_d32_async_invalid_stream() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        assert_eq!(gpu.memset_d32_async(ptr, 0xFF, 4, 0xBAD), Err(CuResult::InvalidValue));
    }

    // --- Primary context tests ---

    #[test]
    fn test_primary_ctx_retain_returns_same_handle() {
        let gpu = StubGpuBackend::new();
        let h1 = gpu.primary_ctx_retain(0).unwrap();
        let h2 = gpu.primary_ctx_retain(0).unwrap();
        assert_eq!(h1, h2, "repeated retain must return the same handle");
        // Cleanup
        let _ = gpu.primary_ctx_release(0);
        let _ = gpu.primary_ctx_release(0);
    }

    #[test]
    fn test_primary_ctx_retain_release_lifecycle() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.primary_ctx_retain(0).unwrap();
        // Retain a second time
        let ctx2 = gpu.primary_ctx_retain(0).unwrap();
        assert_eq!(ctx, ctx2);
        // First release: context still alive
        gpu.primary_ctx_release(0).unwrap();
        assert!(gpu.ctx_exists(ctx), "context should still exist after first release");
        // Second release: refcount hits 0, context destroyed
        gpu.primary_ctx_release(0).unwrap();
        assert!(!gpu.ctx_exists(ctx), "context should be destroyed when refcount reaches 0");
    }

    #[test]
    fn test_primary_ctx_get_state_active_inactive() {
        let gpu = StubGpuBackend::new();
        // No primary ctx yet: inactive
        let (flags, active) = gpu.primary_ctx_get_state(0).unwrap();
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
        // Retain: active
        let _ = gpu.primary_ctx_retain(0).unwrap();
        let (_, active) = gpu.primary_ctx_get_state(0).unwrap();
        assert_eq!(active, 1);
        // Release: inactive again
        gpu.primary_ctx_release(0).unwrap();
        let (_, active) = gpu.primary_ctx_get_state(0).unwrap();
        assert_eq!(active, 0);
    }

    #[test]
    fn test_primary_ctx_set_flags_when_inactive() {
        let gpu = StubGpuBackend::new();
        gpu.primary_ctx_set_flags(0, 0x04).unwrap();
        let (flags, _) = gpu.primary_ctx_get_state(0).unwrap();
        assert_eq!(flags, 0x04);
    }

    #[test]
    fn test_primary_ctx_set_flags_rejected_when_active() {
        let gpu = StubGpuBackend::new();
        let _ = gpu.primary_ctx_retain(0).unwrap();
        assert_eq!(
            gpu.primary_ctx_set_flags(0, 0x04),
            Err(CuResult::PrimaryContextActive)
        );
        let _ = gpu.primary_ctx_release(0);
    }

    #[test]
    fn test_primary_ctx_reset_destroys_regardless_of_refcount() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.primary_ctx_retain(0).unwrap();
        let _ = gpu.primary_ctx_retain(0).unwrap(); // refcount = 2
        let old = gpu.primary_ctx_reset(0).unwrap();
        assert_eq!(old, Some(ctx));
        assert!(!gpu.ctx_exists(ctx), "context should be destroyed after reset");
        let (_, active) = gpu.primary_ctx_get_state(0).unwrap();
        assert_eq!(active, 0, "should be inactive after reset");
    }

    #[test]
    fn test_primary_ctx_reset_preserves_flags() {
        let gpu = StubGpuBackend::new();
        gpu.primary_ctx_set_flags(0, 0x08).unwrap();
        let _ = gpu.primary_ctx_retain(0).unwrap();
        gpu.primary_ctx_reset(0).unwrap();
        let (flags, _) = gpu.primary_ctx_get_state(0).unwrap();
        assert_eq!(flags, 0x08, "flags should be preserved after reset");
    }

    #[test]
    fn test_primary_ctx_invalid_device() {
        let gpu = StubGpuBackend::new();
        assert!(gpu.primary_ctx_retain(99).is_err());
        assert!(gpu.primary_ctx_release(99).is_err());
        assert!(gpu.primary_ctx_get_state(99).is_err());
        assert!(gpu.primary_ctx_set_flags(99, 0).is_err());
        assert!(gpu.primary_ctx_reset(99).is_err());
    }

    #[test]
    fn test_primary_ctx_release_without_retain_fails() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.primary_ctx_release(0), Err(CuResult::InvalidContext));
    }

    #[test]
    fn test_primary_ctx_retain_after_reset_creates_new() {
        let gpu = StubGpuBackend::new();
        let h1 = gpu.primary_ctx_retain(0).unwrap();
        gpu.primary_ctx_reset(0).unwrap();
        let h2 = gpu.primary_ctx_retain(0).unwrap();
        assert_ne!(h1, h2, "new handle should be allocated after reset");
        let _ = gpu.primary_ctx_release(0);
    }

    // --- func_get_attribute tests ---

    /// Helper: create a module + function in the stub and return the func handle.
    fn setup_stub_function(gpu: &StubGpuBackend) -> u64 {
        let module = gpu.module_load_data(b"ptx").unwrap();
        gpu.module_get_function(module, "my_kernel").unwrap()
    }

    #[test]
    fn test_func_get_attribute_max_threads() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_get_attribute(0, func).unwrap(), 1024);
    }

    #[test]
    fn test_func_get_attribute_num_regs() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_get_attribute(4, func).unwrap(), 32);
    }

    #[test]
    fn test_func_get_attribute_ptx_version() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_get_attribute(5, func).unwrap(), 80);
    }

    #[test]
    fn test_func_get_attribute_binary_version() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_get_attribute(6, func).unwrap(), 80);
    }

    #[test]
    fn test_func_get_attribute_shared_size() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_get_attribute(1, func).unwrap(), 0);
    }

    #[test]
    fn test_func_get_attribute_all_valid_attribs() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        // All attribs 0..=15 should succeed
        for attrib in 0..=15 {
            assert!(gpu.func_get_attribute(attrib, func).is_ok(),
                "attrib {} should be valid", attrib);
        }
    }

    #[test]
    fn test_func_get_attribute_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.func_get_attribute(0, 0xDEAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_func_get_attribute_invalid_attrib() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_get_attribute(9999, func), Err(CuResult::InvalidValue));
        assert_eq!(gpu.func_get_attribute(-1, func), Err(CuResult::InvalidValue));
        assert_eq!(gpu.func_get_attribute(16, func), Err(CuResult::InvalidValue));
    }

    // ----- Stream priority/flags/ctx tests -----

    #[test]
    fn test_stream_create_with_priority() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        let stream = gpu.stream_create_with_priority(1, -1, ctx).unwrap();
        assert_ne!(stream, 0);
        // Verify metadata was stored correctly.
        assert_eq!(gpu.stream_get_priority(stream).unwrap(), -1);
        assert_eq!(gpu.stream_get_flags(stream).unwrap(), 1);
        assert_eq!(gpu.stream_get_ctx(stream).unwrap(), ctx);
    }

    #[test]
    fn test_stream_create_with_priority_zero() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create_with_priority(0, 0, 0).unwrap();
        assert_ne!(stream, 0);
        assert_eq!(gpu.stream_get_priority(stream).unwrap(), 0);
        assert_eq!(gpu.stream_get_flags(stream).unwrap(), 0);
        assert_eq!(gpu.stream_get_ctx(stream).unwrap(), 0);
    }

    #[test]
    fn test_stream_get_priority_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.stream_get_priority(0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_stream_get_flags_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.stream_get_flags(0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_stream_get_ctx_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.stream_get_ctx(0xBAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_stream_create_default_has_zero_priority_and_ctx() {
        let gpu = StubGpuBackend::new();
        // stream_create (the basic one) should set priority=0, ctx=0
        let stream = gpu.stream_create(0x01).unwrap();
        assert_eq!(gpu.stream_get_priority(stream).unwrap(), 0);
        assert_eq!(gpu.stream_get_flags(stream).unwrap(), 0x01);
        assert_eq!(gpu.stream_get_ctx(stream).unwrap(), 0);
    }

    #[test]
    fn test_stream_metadata_survives_across_queries() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        let s1 = gpu.stream_create_with_priority(0x02, -5, ctx).unwrap();
        let s2 = gpu.stream_create_with_priority(0x04, 3, ctx).unwrap();
        // Each stream has independent metadata.
        assert_eq!(gpu.stream_get_priority(s1).unwrap(), -5);
        assert_eq!(gpu.stream_get_priority(s2).unwrap(), 3);
        assert_eq!(gpu.stream_get_flags(s1).unwrap(), 0x02);
        assert_eq!(gpu.stream_get_flags(s2).unwrap(), 0x04);
    }

    #[test]
    fn test_stream_create_with_priority_then_destroy() {
        let gpu = StubGpuBackend::new();
        let stream = gpu.stream_create_with_priority(0, -1, 0).unwrap();
        assert!(gpu.stream_destroy(stream).is_ok());
        // After destroy, queries should fail.
        assert_eq!(gpu.stream_get_priority(stream), Err(CuResult::InvalidValue));
        assert_eq!(gpu.stream_get_flags(stream), Err(CuResult::InvalidValue));
        assert_eq!(gpu.stream_get_ctx(stream), Err(CuResult::InvalidValue));
    }

    // --- Context push/pop tests ---

    #[test]
    fn test_ctx_push_pop_roundtrip() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert!(gpu.ctx_push_current(ctx).is_ok());
        let popped = gpu.ctx_pop_current().unwrap();
        assert_eq!(popped, ctx);
    }

    #[test]
    fn test_ctx_push_invalid_context() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_push_current(0xDEAD), Err(CuResult::InvalidContext));
    }

    #[test]
    fn test_ctx_pop_empty_stack() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_pop_current(), Err(CuResult::InvalidContext));
    }

    #[test]
    fn test_ctx_push_pop_lifo_order() {
        let gpu = StubGpuBackend::new();
        let ctx1 = gpu.ctx_create(0, 0).unwrap();
        let ctx2 = gpu.ctx_create(0, 0).unwrap();
        gpu.ctx_push_current(ctx1).unwrap();
        gpu.ctx_push_current(ctx2).unwrap();
        assert_eq!(gpu.ctx_pop_current().unwrap(), ctx2);
        assert_eq!(gpu.ctx_pop_current().unwrap(), ctx1);
    }

    // --- Context API version tests ---

    #[test]
    fn test_ctx_get_api_version() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert_eq!(gpu.ctx_get_api_version(ctx).unwrap(), 12000);
    }

    #[test]
    fn test_ctx_get_api_version_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_get_api_version(0xDEAD), Err(CuResult::InvalidContext));
    }

    // --- Context limits tests ---

    #[test]
    fn test_ctx_get_limit_defaults() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_get_limit(0x00).unwrap(), 1024);       // stack size
        assert_eq!(gpu.ctx_get_limit(0x01).unwrap(), 1_048_576);   // printf FIFO
        assert_eq!(gpu.ctx_get_limit(0x02).unwrap(), 8_388_608);   // malloc heap
    }

    #[test]
    fn test_ctx_get_limit_unknown() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_get_limit(0xFF), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_ctx_set_limit() {
        let gpu = StubGpuBackend::new();
        gpu.ctx_set_limit(0x00, 2048).unwrap();
        assert_eq!(gpu.ctx_get_limit(0x00).unwrap(), 2048);
    }

    #[test]
    fn test_ctx_set_limit_new_key() {
        let gpu = StubGpuBackend::new();
        gpu.ctx_set_limit(0x05, 12345).unwrap();
        assert_eq!(gpu.ctx_get_limit(0x05).unwrap(), 12345);
    }

    // --- Stream priority range tests ---

    #[test]
    fn test_ctx_get_stream_priority_range() {
        let gpu = StubGpuBackend::new();
        let (least, greatest) = gpu.ctx_get_stream_priority_range().unwrap();
        assert_eq!(least, 0);
        assert_eq!(greatest, -1);
    }

    // --- Context flags tests ---

    #[test]
    fn test_ctx_get_flags() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0x01, 0).unwrap();
        assert_eq!(gpu.ctx_get_flags(ctx).unwrap(), 0x01);
    }

    #[test]
    fn test_ctx_get_flags_default() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert_eq!(gpu.ctx_get_flags(ctx).unwrap(), 0);
    }

    #[test]
    fn test_ctx_get_flags_invalid_context() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_get_flags(0xDEAD), Err(CuResult::InvalidContext));
    }

    // --- Occupancy tests ---

    /// Helper: create a module + function in the stub.
    fn setup_func(gpu: &StubGpuBackend) -> u64 {
        let module = gpu.module_load_data(b"ptx").unwrap();
        gpu.module_get_function(module, "kern").unwrap()
    }

    #[test]
    fn test_occupancy_max_active_blocks_basic() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        // block_size=256 => 2048/256=8, min(8,16)=8
        let blocks = gpu.occupancy_max_active_blocks(func, 256, 0, 0).unwrap();
        assert_eq!(blocks, 8);
    }

    #[test]
    fn test_occupancy_max_active_blocks_large_block() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        // block_size=1024 => 2048/1024=2, min(2,16)=2
        let blocks = gpu.occupancy_max_active_blocks(func, 1024, 0, 0).unwrap();
        assert_eq!(blocks, 2);
    }

    #[test]
    fn test_occupancy_max_active_blocks_small_block() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        // block_size=32 => 2048/32=64, min(64,16)=16
        let blocks = gpu.occupancy_max_active_blocks(func, 32, 0, 0).unwrap();
        assert_eq!(blocks, 16);
    }

    #[test]
    fn test_occupancy_max_active_blocks_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(
            gpu.occupancy_max_active_blocks(0xDEAD, 256, 0, 0),
            Err(CuResult::InvalidValue)
        );
    }

    #[test]
    fn test_occupancy_max_active_blocks_zero_block_size() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        assert_eq!(
            gpu.occupancy_max_active_blocks(func, 0, 0, 0),
            Err(CuResult::InvalidValue)
        );
    }

    #[test]
    fn test_occupancy_max_active_blocks_negative_block_size() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        assert_eq!(
            gpu.occupancy_max_active_blocks(func, -1, 0, 0),
            Err(CuResult::InvalidValue)
        );
    }

    #[test]
    fn test_occupancy_max_potential_block_size_basic() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        let (min_grid, block_sz) = gpu.occupancy_max_potential_block_size(func, 0, 0, 0).unwrap();
        assert_eq!(block_sz, 256);
        // 2048/256=8 blocks/SM * 82 SMs = 656
        assert_eq!(min_grid, 656);
    }

    #[test]
    fn test_occupancy_max_potential_block_size_with_limit() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        let (min_grid, block_sz) = gpu.occupancy_max_potential_block_size(func, 0, 128, 0).unwrap();
        assert_eq!(block_sz, 128);
        // 2048/128=16 blocks/SM * 82 SMs = 1312
        assert_eq!(min_grid, 1312);
    }

    #[test]
    fn test_occupancy_max_potential_block_size_large_limit() {
        let gpu = StubGpuBackend::new();
        let func = setup_func(&gpu);
        // limit > 256 => uses default 256
        let (_, block_sz) = gpu.occupancy_max_potential_block_size(func, 0, 512, 0).unwrap();
        assert_eq!(block_sz, 256);
    }

    #[test]
    fn test_occupancy_max_potential_block_size_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(
            gpu.occupancy_max_potential_block_size(0xDEAD, 0, 0, 0),
            Err(CuResult::InvalidValue)
        );
    }

    // --- Peer access tests ---

    #[test]
    fn test_device_can_access_peer_same_device() {
        let gpu = StubGpuBackend::new();
        // Per CUDA spec: self-peer returns 0
        assert_eq!(gpu.device_can_access_peer(0, 0).unwrap(), 0);
    }

    #[test]
    fn test_device_can_access_peer_invalid_device() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.device_can_access_peer(0, 99), Err(CuResult::InvalidDevice));
        assert_eq!(gpu.device_can_access_peer(99, 0), Err(CuResult::InvalidDevice));
    }

    #[test]
    fn test_device_get_p2p_attribute_returns_zero() {
        let gpu = StubGpuBackend::new();
        // All P2P attributes return 0 in stub
        assert_eq!(gpu.device_get_p2p_attribute(0, 0, 0).unwrap(), 0); // PERFORMANCE_RANK
        assert_eq!(gpu.device_get_p2p_attribute(1, 0, 0).unwrap(), 0); // ACCESS_SUPPORTED
        assert_eq!(gpu.device_get_p2p_attribute(2, 0, 0).unwrap(), 0); // NATIVE_ATOMIC_SUPPORTED
        assert_eq!(gpu.device_get_p2p_attribute(3, 0, 0).unwrap(), 0); // CUDA_ARRAY_ACCESS_SUPPORTED
    }

    #[test]
    fn test_device_get_p2p_attribute_invalid_device() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.device_get_p2p_attribute(0, 99, 0), Err(CuResult::InvalidDevice));
        assert_eq!(gpu.device_get_p2p_attribute(0, 0, 99), Err(CuResult::InvalidDevice));
    }

    #[test]
    fn test_ctx_enable_peer_access() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert!(gpu.ctx_enable_peer_access(ctx, 0).is_ok());
    }

    #[test]
    fn test_ctx_enable_peer_access_already_enabled() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        gpu.ctx_enable_peer_access(ctx, 0).unwrap();
        assert_eq!(gpu.ctx_enable_peer_access(ctx, 0), Err(CuResult::PeerAccessAlreadyEnabled));
    }

    #[test]
    fn test_ctx_enable_peer_access_invalid_context() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_enable_peer_access(0xDEAD, 0), Err(CuResult::InvalidContext));
    }

    #[test]
    fn test_ctx_disable_peer_access() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        gpu.ctx_enable_peer_access(ctx, 0).unwrap();
        assert!(gpu.ctx_disable_peer_access(ctx).is_ok());
    }

    #[test]
    fn test_ctx_disable_peer_access_not_enabled() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert_eq!(gpu.ctx_disable_peer_access(ctx), Err(CuResult::PeerAccessNotEnabled));
    }

    #[test]
    fn test_ctx_disable_peer_access_double_disable() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        gpu.ctx_enable_peer_access(ctx, 0).unwrap();
        gpu.ctx_disable_peer_access(ctx).unwrap();
        assert_eq!(gpu.ctx_disable_peer_access(ctx), Err(CuResult::PeerAccessNotEnabled));
    }

    // --- func_set_attribute tests ---

    #[test]
    fn test_func_set_attribute_max_dynamic_shared_mem() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        // attrib 8 = MAX_DYNAMIC_SHARED_SIZE_BYTES
        assert!(gpu.func_set_attribute(func, 8, 65536).is_ok());
        // Verify it's readable back
        assert_eq!(gpu.func_get_attribute(8, func).unwrap(), 65536);
    }

    #[test]
    fn test_func_set_attribute_preferred_carveout() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        // attrib 9 = PREFERRED_SHARED_MEMORY_CARVEOUT
        assert!(gpu.func_set_attribute(func, 9, 50).is_ok());
        assert_eq!(gpu.func_get_attribute(9, func).unwrap(), 50);
    }

    #[test]
    fn test_func_set_attribute_read_only_rejected() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        // attrib 0 = MAX_THREADS_PER_BLOCK (read-only)
        assert_eq!(gpu.func_set_attribute(func, 0, 512), Err(CuResult::InvalidValue));
        // attrib 4 = NUM_REGS (read-only)
        assert_eq!(gpu.func_set_attribute(func, 4, 16), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_func_set_attribute_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.func_set_attribute(0xDEAD, 8, 100), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_func_set_attribute_invalid_attrib() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        assert_eq!(gpu.func_set_attribute(func, 9999, 0), Err(CuResult::InvalidValue));
        assert_eq!(gpu.func_set_attribute(func, -1, 0), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_func_set_attribute_overwrite() {
        let gpu = StubGpuBackend::new();
        let func = setup_stub_function(&gpu);
        gpu.func_set_attribute(func, 8, 1024).unwrap();
        assert_eq!(gpu.func_get_attribute(8, func).unwrap(), 1024);
        gpu.func_set_attribute(func, 8, 2048).unwrap();
        assert_eq!(gpu.func_get_attribute(8, func).unwrap(), 2048);
    }

    // --- mem_get_address_range tests ---

    #[test]
    fn test_mem_get_address_range_exact_match() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(4096).unwrap();
        let (base, size) = gpu.mem_get_address_range(ptr).unwrap();
        assert_eq!(base, ptr);
        assert_eq!(size, 4096);
    }

    #[test]
    fn test_mem_get_address_range_unknown_ptr() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.mem_get_address_range(0xDEAD), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_mem_get_address_range_after_free() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(1024).unwrap();
        gpu.mem_free(ptr).unwrap();
        assert_eq!(gpu.mem_get_address_range(ptr), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_mem_get_address_range_interior_ptr_fails() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(4096).unwrap();
        // Interior pointer (ptr + 100) is not the base, stub doesn't support this
        assert_eq!(gpu.mem_get_address_range(ptr + 100), Err(CuResult::InvalidValue));
    }

    // --- Pointer attribute tests ---

    #[test]
    fn test_pointer_get_attribute_memory_type_device() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(1024).unwrap();
        let val = gpu.pointer_get_attribute(2, ptr).unwrap(); // MEMORY_TYPE
        assert_eq!(val, 2); // CU_MEMORYTYPE_DEVICE
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attribute_memory_type_host() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc_host(1024).unwrap();
        let val = gpu.pointer_get_attribute(2, ptr).unwrap(); // MEMORY_TYPE
        assert_eq!(val, 1); // CU_MEMORYTYPE_HOST
        let _ = gpu.mem_free_host(ptr);
    }

    #[test]
    fn test_pointer_get_attribute_memory_type_unknown() {
        let gpu = StubGpuBackend::new();
        let result = gpu.pointer_get_attribute(2, 0xDEAD_BEEF);
        assert_eq!(result, Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_pointer_get_attribute_device_pointer() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(512).unwrap();
        let val = gpu.pointer_get_attribute(3, ptr).unwrap(); // DEVICE_POINTER
        assert_eq!(val, ptr);
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attribute_host_pointer() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(512).unwrap();
        let val = gpu.pointer_get_attribute(4, ptr).unwrap(); // HOST_POINTER
        assert_eq!(val, 0); // device pointers have no host pointer
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attribute_context() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(256).unwrap();
        let val = gpu.pointer_get_attribute(1, ptr).unwrap(); // CONTEXT
        assert_eq!(val, 0); // stub doesn't track allocation context
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attribute_is_managed() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(256).unwrap();
        let val = gpu.pointer_get_attribute(6, ptr).unwrap(); // IS_MANAGED
        assert_eq!(val, 0); // no managed memory in stub
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attribute_device_ordinal() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(256).unwrap();
        let val = gpu.pointer_get_attribute(8, ptr).unwrap(); // DEVICE_ORDINAL
        assert_eq!(val, 0);
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attributes_multiple() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(1024).unwrap();
        // Query MEMORY_TYPE(2) and DEVICE_POINTER(3) and DEVICE_ORDINAL(8)
        let attrs = vec![2, 3, 8];
        let vals = gpu.pointer_get_attributes(&attrs, ptr).unwrap();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0], 2); // DEVICE
        assert_eq!(vals[1], ptr); // device pointer value
        assert_eq!(vals[2], 0); // ordinal 0
        let _ = gpu.mem_free(ptr);
    }

    #[test]
    fn test_pointer_get_attributes_unknown_ptr() {
        let gpu = StubGpuBackend::new();
        let result = gpu.pointer_get_attributes(&[2], 0xDEAD);
        assert_eq!(result, Err(CuResult::InvalidValue));
    }

    // --- Cache config tests ---

    #[test]
    fn test_ctx_get_cache_config_default() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_get_cache_config().unwrap(), 0);
    }

    #[test]
    fn test_ctx_set_cache_config() {
        let gpu = StubGpuBackend::new();
        gpu.ctx_set_cache_config(0x01).unwrap(); // PREFER_SHARED
        assert_eq!(gpu.ctx_get_cache_config().unwrap(), 0x01);
        gpu.ctx_set_cache_config(0x02).unwrap(); // PREFER_L1
        assert_eq!(gpu.ctx_get_cache_config().unwrap(), 0x02);
        gpu.ctx_set_cache_config(0x03).unwrap(); // PREFER_EQUAL
        assert_eq!(gpu.ctx_get_cache_config().unwrap(), 0x03);
    }

    #[test]
    fn test_ctx_set_cache_config_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_set_cache_config(0x04), Err(CuResult::InvalidValue));
    }

    // --- Shared mem config tests ---

    #[test]
    fn test_ctx_get_shared_mem_config_default() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_get_shared_mem_config().unwrap(), 0);
    }

    #[test]
    fn test_ctx_set_shared_mem_config() {
        let gpu = StubGpuBackend::new();
        gpu.ctx_set_shared_mem_config(0x01).unwrap(); // FOUR_BYTE_BANK_SIZE
        assert_eq!(gpu.ctx_get_shared_mem_config().unwrap(), 0x01);
        gpu.ctx_set_shared_mem_config(0x02).unwrap(); // EIGHT_BYTE_BANK_SIZE
        assert_eq!(gpu.ctx_get_shared_mem_config().unwrap(), 0x02);
    }

    #[test]
    fn test_ctx_set_shared_mem_config_invalid() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.ctx_set_shared_mem_config(0x03), Err(CuResult::InvalidValue));
    }

    // --- FuncSetCacheConfig tests ---

    #[test]
    fn test_func_set_cache_config() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "my_kernel").unwrap();
        gpu.func_set_cache_config(func, 0x02).unwrap(); // PREFER_L1
    }

    #[test]
    fn test_func_set_cache_config_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.func_set_cache_config(0xDEAD, 0x01), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_func_set_cache_config_invalid_config() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "my_kernel").unwrap();
        assert_eq!(gpu.func_set_cache_config(func, 0x04), Err(CuResult::InvalidValue));
    }

    // --- FuncSetSharedMemConfig tests ---

    #[test]
    fn test_func_set_shared_mem_config() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "my_kernel").unwrap();
        gpu.func_set_shared_mem_config(func, 0x01).unwrap(); // FOUR_BYTE
    }

    #[test]
    fn test_func_set_shared_mem_config_invalid_func() {
        let gpu = StubGpuBackend::new();
        assert_eq!(gpu.func_set_shared_mem_config(0xDEAD, 0x01), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_func_set_shared_mem_config_invalid_config() {
        let gpu = StubGpuBackend::new();
        let module = gpu.module_load_data(b"ptx").unwrap();
        let func = gpu.module_get_function(module, "my_kernel").unwrap();
        assert_eq!(gpu.func_set_shared_mem_config(func, 0x03), Err(CuResult::InvalidValue));
    }
}
