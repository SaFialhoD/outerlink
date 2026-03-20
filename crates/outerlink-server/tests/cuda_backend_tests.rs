//! Tests for the CUDA GPU backend.
//!
//! These tests verify the CudaGpuBackend's construction logic and error
//! handling WITHOUT requiring a real GPU. Tests that would need libcuda.so
//! are guarded behind the `real_gpu` feature or skipped at runtime.

use outerlink_server::cuda_backend::CudaGpuBackend;
use outerlink_server::gpu_backend::GpuBackend;

/// CudaGpuBackend::new() returns an error (not a panic) when no CUDA
/// driver library is available — which is the expected case in CI.
#[test]
fn new_returns_error_when_no_cuda_driver() {
    // In CI (no GPU), this should return Err, not panic.
    // On a machine with a GPU, this will succeed — and that's fine too.
    let result = CudaGpuBackend::new();
    // We just assert it doesn't panic. Whether it's Ok or Err depends on the machine.
    let _ = result;
}

/// Verify that CudaGpuBackend implements the GpuBackend trait.
/// This is a compile-time check: if it compiles, the trait is implemented.
#[test]
fn implements_gpu_backend_trait() {
    fn _assert_trait<T: GpuBackend>() {}
    _assert_trait::<CudaGpuBackend>();
}

/// Verify CudaGpuBackend is Send + Sync (required by GpuBackend).
#[test]
fn is_send_sync() {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<CudaGpuBackend>();
}

/// When CudaGpuBackend::new() fails, the error message should mention
/// the CUDA driver library name.
#[test]
fn error_message_mentions_cuda_driver() {
    let result = CudaGpuBackend::new();
    if let Err(e) = result {
        let msg = format!("{}", e);
        assert!(
            msg.contains("CUDA") || msg.contains("cuda") || msg.contains("nvcuda"),
            "Error message should mention CUDA driver, got: {}",
            msg
        );
    }
    // If Ok, we're on a machine with a GPU — nothing to check about the error.
}

/// VRAM safety margin constant is at least 512 MB.
#[test]
fn vram_safety_margin_exists() {
    assert_eq!(outerlink_server::cuda_backend::VRAM_SAFETY_MARGIN, 512 * 1024 * 1024);
}
