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
    fn mem_free(&self, ptr: u64) -> CuResult;

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

    /// Set the current CUDA context for the calling thread.
    fn ctx_set_current(&self, ctx: u64) -> Result<(), CuResult>;

    /// Get the current CUDA context.
    fn ctx_get_current(&self) -> Result<u64, CuResult>;

    /// Get the device ordinal for a context.
    fn ctx_get_device(&self, ctx: u64) -> Result<i32, CuResult>;

    /// Synchronize the current context (block until all work completes).
    fn ctx_synchronize(&self) -> Result<(), CuResult>;
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
    /// The current context handle (0 = none).
    current_ctx: u64,
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
                current_ctx: 0,
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

    fn mem_free(&self, ptr: u64) -> CuResult {
        match self.state.lock().unwrap().allocations.remove(&ptr) {
            Some(_) => CuResult::Success,
            None => CuResult::InvalidValue,
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
        state.current_ctx = id;
        Ok(id)
    }

    fn ctx_destroy(&self, ctx: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if state.contexts.remove(&ctx).is_none() {
            return Err(CuResult::InvalidContext);
        }
        if state.current_ctx == ctx {
            state.current_ctx = 0;
        }
        Ok(())
    }

    fn ctx_set_current(&self, ctx: u64) -> Result<(), CuResult> {
        let mut state = self.state.lock().unwrap();
        if ctx == 0 {
            state.current_ctx = 0;
            return Ok(());
        }
        if !state.contexts.contains_key(&ctx) {
            return Err(CuResult::InvalidContext);
        }
        state.current_ctx = ctx;
        Ok(())
    }

    fn ctx_get_current(&self) -> Result<u64, CuResult> {
        let state = self.state.lock().unwrap();
        Ok(state.current_ctx)
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
        assert_eq!(gpu.mem_free(ptr), CuResult::Success);
        // Double free should fail.
        assert_eq!(gpu.mem_free(ptr), CuResult::InvalidValue);
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

        gpu.mem_free(ptr);
    }

    #[test]
    fn test_memcpy_htod_overflow() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        let big = vec![0u8; 32];
        assert_eq!(gpu.memcpy_htod(ptr, &big), CuResult::InvalidValue);
        gpu.mem_free(ptr);
    }

    #[test]
    fn test_memcpy_dtoh_overflow() {
        let gpu = StubGpuBackend::new();
        let ptr = gpu.mem_alloc(16).unwrap();
        assert_eq!(gpu.memcpy_dtoh(ptr, 32), Err(CuResult::InvalidValue));
        gpu.mem_free(ptr);
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

        gpu.mem_free(ptr);
    }

    #[test]
    fn test_ctx_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let ctx = gpu.ctx_create(0, 0).unwrap();
        assert_ne!(ctx, 0);
        // Should be set as current.
        assert_eq!(gpu.ctx_get_current().unwrap(), ctx);
        // Destroy it.
        assert!(gpu.ctx_destroy(ctx).is_ok());
        // Current should now be cleared.
        assert_eq!(gpu.ctx_get_current().unwrap(), 0);
    }

    #[test]
    fn test_ctx_set_current() {
        let gpu = StubGpuBackend::new();
        let ctx1 = gpu.ctx_create(0, 0).unwrap();
        let ctx2 = gpu.ctx_create(0, 0).unwrap();
        // ctx2 should be current (most recently created).
        assert_eq!(gpu.ctx_get_current().unwrap(), ctx2);
        // Switch to ctx1.
        assert!(gpu.ctx_set_current(ctx1).is_ok());
        assert_eq!(gpu.ctx_get_current().unwrap(), ctx1);
        // Unset current (ctx=0).
        assert!(gpu.ctx_set_current(0).is_ok());
        assert_eq!(gpu.ctx_get_current().unwrap(), 0);
        // Invalid context should error.
        assert_eq!(gpu.ctx_set_current(0xBAAD), Err(CuResult::InvalidContext));
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
}
