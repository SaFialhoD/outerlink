//! NVML virtualization layer.
//!
//! Provides virtual GPU monitoring data (temperature, power, clocks, memory,
//! utilization) from remote OuterLink servers. When no server is connected,
//! returns stub data matching an RTX 3090 configuration.
//!
//! The [`NvmlVirtualizer`] holds cached [`NvmlGpuSnapshot`]s and is accessed
//! by the FFI functions in `nvml_ffi.rs`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use outerlink_common::nvml_types::NvmlGpuSnapshot;

/// Global NVML virtualizer singleton.
static NVML_VIRTUALIZER: OnceLock<NvmlVirtualizer> = OnceLock::new();

/// Get a reference to the global NVML virtualizer (initializes on first call).
pub fn nvml_virtualizer() -> &'static NvmlVirtualizer {
    NVML_VIRTUALIZER.get_or_init(NvmlVirtualizer::new)
}

/// Manages cached NVML GPU snapshots from the remote server.
pub struct NvmlVirtualizer {
    /// Cached GPU snapshots from the server (or stubs).
    snapshots: RwLock<Vec<NvmlGpuSnapshot>>,
    /// Whether `nvmlInit` has been called.
    initialized: AtomicBool,
}

impl NvmlVirtualizer {
    /// Create a new virtualizer (not yet initialized).
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(Vec::new()),
            initialized: AtomicBool::new(false),
        }
    }

    /// Initialize the NVML virtualizer.
    ///
    /// Populates snapshots with stub data for now. When server connectivity
    /// is wired up, this will request real snapshots via the protocol.
    ///
    /// Thread-safe: takes the write lock before checking/setting `initialized`,
    /// avoiding TOCTOU races. Supports shutdown+reinit cycles.
    pub fn init(&self) {
        // Take the write lock FIRST to prevent TOCTOU: two threads could both
        // pass an atomic load check before either acquires the lock.
        let mut snaps = self.snapshots.write().unwrap();
        if self.initialized.load(Ordering::Acquire) {
            return; // Already initialized (checked under lock)
        }

        // Populate with stub data (1 GPU matching RTX 3090).
        // When connected to a server, this will be replaced by real data.
        snaps.push(NvmlGpuSnapshot::stub_rtx3090(0));
        self.initialized.store(true, Ordering::Release);
    }

    /// Shut down the NVML virtualizer.
    pub fn shutdown(&self) {
        self.initialized.store(false, Ordering::Release);
        let mut snaps = self.snapshots.write().unwrap();
        snaps.clear();
    }

    /// Check whether the virtualizer has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Return the number of virtual GPU devices.
    pub fn device_count(&self) -> u32 {
        let snaps = self.snapshots.read().unwrap();
        snaps.len() as u32
    }

    /// Get a snapshot by device index.
    pub fn get_snapshot(&self, index: u32) -> Option<NvmlGpuSnapshot> {
        let snaps = self.snapshots.read().unwrap();
        snaps.get(index as usize).cloned()
    }

    /// Get a snapshot by UUID string.
    pub fn get_snapshot_by_uuid(&self, uuid: &str) -> Option<(u32, NvmlGpuSnapshot)> {
        let snaps = self.snapshots.read().unwrap();
        for (i, snap) in snaps.iter().enumerate() {
            if snap.uuid_str() == uuid {
                return Some((i as u32, snap.clone()));
            }
        }
        None
    }

    /// Refresh snapshots from the server.
    ///
    /// Currently a no-op for stub mode. When server connectivity is wired up,
    /// this will send an `NvmlSnapshotRequest` and update the cache.
    pub fn refresh(&self) {
        // Future: send NvmlSnapshotRequest to server and update snapshots.
    }
}

// NvmlVirtualizer is automatically Send + Sync because RwLock<Vec<T>>
// and AtomicBool are both Send + Sync. No unsafe impl needed.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtualizer_init() {
        let virt = NvmlVirtualizer::new();
        assert!(!virt.is_initialized());
        virt.init();
        assert!(virt.is_initialized());
        assert!(virt.device_count() > 0);
    }

    #[test]
    fn test_virtualizer_double_init_is_safe() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        let count = virt.device_count();
        virt.init(); // Should not add more devices
        assert_eq!(virt.device_count(), count);
    }

    #[test]
    fn test_device_count() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        assert_eq!(virt.device_count(), 1);
    }

    #[test]
    fn test_snapshot_lookup() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        let snap = virt.get_snapshot(0).expect("device 0 should exist");
        assert_eq!(snap.name_str(), "NVIDIA GeForce RTX 3090");
        assert_eq!(snap.compute_cap_major, 8);
        assert_eq!(snap.compute_cap_minor, 6);
    }

    #[test]
    fn test_snapshot_out_of_range() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        assert!(virt.get_snapshot(99).is_none());
    }

    #[test]
    fn test_uuid_lookup() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        let snap = virt.get_snapshot(0).unwrap();
        let uuid = snap.uuid_str().to_string();
        let (idx, found) = virt.get_snapshot_by_uuid(&uuid).expect("uuid lookup should work");
        assert_eq!(idx, 0);
        assert_eq!(found.name_str(), snap.name_str());
    }

    #[test]
    fn test_uuid_lookup_not_found() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        assert!(virt.get_snapshot_by_uuid("GPU-nonexistent").is_none());
    }

    #[test]
    fn test_uninitialized_error() {
        let virt = NvmlVirtualizer::new();
        assert!(!virt.is_initialized());
        // device_count returns 0 when not initialized (no snapshots loaded)
        assert_eq!(virt.device_count(), 0);
    }

    #[test]
    fn test_shutdown() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        assert!(virt.is_initialized());
        virt.shutdown();
        assert!(!virt.is_initialized());
        assert_eq!(virt.device_count(), 0);
    }

    #[test]
    fn test_reinit_after_shutdown() {
        let virt = NvmlVirtualizer::new();
        virt.init();
        virt.shutdown();
        virt.init();
        assert!(virt.is_initialized());
        assert_eq!(virt.device_count(), 1);
    }
}
