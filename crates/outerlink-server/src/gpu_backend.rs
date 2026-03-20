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
}

// ---------------------------------------------------------------------------
// Stub implementation
// ---------------------------------------------------------------------------

/// Total simulated VRAM (24 GiB).
const STUB_TOTAL_MEM: usize = 24 * 1024 * 1024 * 1024;

/// Simulated GPU name.
const STUB_GPU_NAME: &str = "OuterLink Virtual GPU";

/// A fake GPU backend that stores "device memory" in a `HashMap`.
///
/// This is intentionally **not** async -- every method returns instantly.
/// Useful for:
/// * Unit/integration tests that run without hardware.
/// * Validating the protocol and handler logic end-to-end.
pub struct StubGpuBackend {
    /// Simulated VRAM: maps device-pointer -> byte buffer.
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    /// Monotonically increasing counter for fake device pointers.
    next_ptr: Mutex<u64>,
}

impl StubGpuBackend {
    /// Create a new stub backend with no allocations.
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            // Start at a recognisable fake base address.
            next_ptr: Mutex::new(0x0000_DEAD_0000_0000),
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

    /// Return how many bytes are currently "allocated".
    fn used_bytes(&self) -> usize {
        self.allocations
            .lock()
            .unwrap()
            .values()
            .map(|v| v.len())
            .sum()
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
        // Hold allocations lock for the entire check-and-insert to prevent a
        // TOCTOU race where two concurrent callers both pass the capacity check
        // and together exceed STUB_TOTAL_MEM.
        let mut allocs = self.allocations.lock().unwrap();
        let used: usize = allocs.values().map(|v| v.len()).sum();
        if used + size > STUB_TOTAL_MEM {
            return Err(CuResult::OutOfMemory);
        }
        let mut next = self.next_ptr.lock().unwrap();
        let ptr = *next;
        *next += size as u64;
        allocs.insert(ptr, vec![0u8; size]);
        Ok(ptr)
    }

    fn mem_free(&self, ptr: u64) -> CuResult {
        match self.allocations.lock().unwrap().remove(&ptr) {
            Some(_) => CuResult::Success,
            None => CuResult::InvalidValue,
        }
    }

    fn memcpy_htod(&self, dst: u64, data: &[u8]) -> CuResult {
        let mut allocs = self.allocations.lock().unwrap();
        match allocs.get_mut(&dst) {
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
        let allocs = self.allocations.lock().unwrap();
        match allocs.get(&src) {
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
        let used = self.used_bytes();
        Ok((STUB_TOTAL_MEM - used, STUB_TOTAL_MEM))
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
}
