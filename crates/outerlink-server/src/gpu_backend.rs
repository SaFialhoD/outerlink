//! GPU backend trait and stub implementation.
//!
//! The `GpuBackend` trait abstracts all GPU operations so the server can
//! run against a real CUDA driver **or** a stub that fakes everything in
//! memory. The stub is used for testing, CI, and development on machines
//! without a GPU.

use std::collections::HashMap;
use std::sync::Mutex;

use outerlink_common::cuda_types::CuResult;

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

    /// Unload a previously loaded module.
    fn module_unload(&self, module: u64) -> Result<(), CuResult>;

    /// Get a function handle from a loaded module by name.
    fn module_get_function(&self, module: u64, name: &str) -> Result<u64, CuResult>;

    /// Get a global variable from a loaded module by name.
    /// Returns `(device_pointer, size_in_bytes)`.
    fn module_get_global(&self, module: u64, name: &str) -> Result<(u64, usize), CuResult>;

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
}

/// Metadata for a stub CUDA stream.
struct StubStream {
    flags: u32,
}

/// Metadata for a stub CUDA event.
struct StubEvent {
    flags: u32,
    recorded: bool,
    timestamp_ns: u64,
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

    // --- Stream operations ---

    fn stream_create(&self, flags: u32) -> Result<u64, CuResult> {
        let mut state = self.state.lock().unwrap();
        let id = state.next_stream_id;
        state.next_stream_id += 1;
        state.streams.insert(id, StubStream { flags });
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
            return Err(CuResult::InvalidValue);
        }
        if end_ev.timestamp_ns < start_ev.timestamp_ns {
            return Err(CuResult::InvalidValue);
        }
        // Convert nanosecond difference to milliseconds.
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
        // Real CUDA returns an error for this.
        assert_eq!(gpu.event_elapsed_time(e2, e1), Err(CuResult::InvalidValue));
    }

    #[test]
    fn test_event_elapsed_time_not_recorded() {
        let gpu = StubGpuBackend::new();
        let e1 = gpu.event_create(0).unwrap();
        let e2 = gpu.event_create(0).unwrap();
        // Neither recorded yet.
        assert_eq!(gpu.event_elapsed_time(e1, e2), Err(CuResult::InvalidValue));
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
}
