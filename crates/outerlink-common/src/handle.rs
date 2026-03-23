//! Handle translation between local (synthetic) and remote (real) CUDA handles.
//!
//! The client generates synthetic handles for the application and maintains
//! bidirectional mappings to the real handles on the remote server.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Prefix ranges for synthetic handle types (high byte).
/// Makes handles debuggable - you can tell the type from the value.
///
/// These are public so that the client can detect synthetic handles in kernel
/// parameters and translate them before sending to the server.
pub const CONTEXT_PREFIX: u64 = 0x0C00_0000_0000_0000;
pub const DEVICEPTR_PREFIX: u64 = 0x0D00_0000_0000_0000;
pub const MODULE_PREFIX: u64 = 0x0E00_0000_0000_0000;
pub const FUNCTION_PREFIX: u64 = 0x0F00_0000_0000_0000;
pub const STREAM_PREFIX: u64 = 0x1000_0000_0000_0000;
pub const EVENT_PREFIX: u64 = 0x1100_0000_0000_0000;
pub const MEMPOOL_PREFIX: u64 = 0x1200_0000_0000_0000;
pub const LINKSTATE_PREFIX: u64 = 0x1300_0000_0000_0000;
pub const LIBRARY_PREFIX: u64 = 0x1400_0000_0000_0000;
pub const KERNEL_PREFIX: u64 = 0x1500_0000_0000_0000;
pub const GRAPH_PREFIX: u64 = 0x1600_0000_0000_0000;
pub const GRAPH_EXEC_PREFIX: u64 = 0x1700_0000_0000_0000;

/// Mask for detecting synthetic handles. All OuterLink prefixes live in the
/// 0x0C00..0x1700 range of the high byte, so any value with bits set in the
/// top 8 bits (above 0x0BFF_FFFF_FFFF_FFFF) is synthetic. Real NVIDIA GPU
/// virtual addresses never reach this exabyte range.
pub const PREFIX_MASK: u64 = 0xFF00_0000_0000_0000;

/// Minimum prefix value for synthetic handles.
const MIN_SYNTHETIC_PREFIX: u64 = CONTEXT_PREFIX; // 0x0C00...

/// Check whether a 64-bit value looks like an OuterLink synthetic handle.
///
/// Returns true if the high byte falls within the synthetic handle range
/// (0x0C..0x17). Zero is never synthetic.
pub fn is_synthetic_handle(value: u64) -> bool {
    let prefix = value & PREFIX_MASK;
    prefix >= MIN_SYNTHETIC_PREFIX && prefix <= GRAPH_EXEC_PREFIX
}

/// Scan serialized kernel parameters and translate any embedded synthetic
/// device pointers to their real remote addresses.
///
/// The params buffer has the wire format produced by the client FFI:
///   `[4B num_params: u32 LE]`
///   `[4B size_0: u32 LE][size_0 bytes of data]`
///   `[4B size_1: u32 LE][size_1 bytes of data]`
///   ...
///
/// For every 8-byte parameter whose value matches a known device-pointer
/// handle, the value is replaced in-place with the real remote address.
///
/// This is safe because real GPU virtual addresses (typically < 1 TB) never
/// collide with synthetic prefixes (>= 0x0C00_0000_0000_0000).
pub fn translate_device_ptrs_in_params(params: &mut [u8], handles: &HandleStore) {
    if params.len() < 4 {
        return;
    }
    let num_params = u32::from_le_bytes(params[..4].try_into().unwrap()) as usize;
    let mut offset = 4;

    for _ in 0..num_params {
        if offset + 4 > params.len() {
            break;
        }
        let size = u32::from_le_bytes(params[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if size == 8 && offset + 8 <= params.len() {
            let value = u64::from_le_bytes(params[offset..offset + 8].try_into().unwrap());
            if is_synthetic_handle(value) {
                // Try device_ptrs first (most common in kernel params)
                if let Some(remote) = handles.device_ptrs.to_remote(value) {
                    params[offset..offset + 8].copy_from_slice(&remote.to_le_bytes());
                }
                // Could also be other handle types passed as kernel args
                // (e.g., stream handles). Extend here if needed.
            }
        }

        offset += size;
    }
}

/// Thread-safe bidirectional handle mapping.
///
/// Maps synthetic local handles <-> real remote handles.
/// Uses DashMap (sharded concurrent HashMap) for lock-free reads.
pub struct HandleMap {
    /// Local synthetic -> remote real
    local_to_remote: DashMap<u64, u64>,
    /// Remote real -> local synthetic
    remote_to_local: DashMap<u64, u64>,
    /// Counter for generating unique synthetic handles
    next_id: AtomicU64,
    /// Prefix for this handle type
    prefix: u64,
}

impl HandleMap {
    /// Create a new handle map with the given type prefix.
    pub fn new(prefix: u64) -> Self {
        Self {
            local_to_remote: DashMap::new(),
            remote_to_local: DashMap::new(),
            next_id: AtomicU64::new(1),
            prefix,
        }
    }

    /// Allocate a new synthetic local handle and map it to a remote handle.
    /// Thread-safe: uses DashMap entry API to avoid TOCTOU race.
    pub fn insert(&self, remote: u64) -> u64 {
        use dashmap::mapref::entry::Entry;
        match self.remote_to_local.entry(remote) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let local = self.prefix | id;
                e.insert(local);
                self.local_to_remote.insert(local, remote);
                local
            }
        }
    }

    /// Return the existing local handle for a remote, or create a new mapping.
    ///
    /// This is semantically identical to [`insert`] but makes the intent
    /// explicit: primary contexts always return the same handle, so repeated
    /// calls with the same remote must yield the same local.
    pub fn insert_or_get(&self, remote: u64) -> u64 {
        self.insert(remote)
    }

    /// Look up the remote handle for a local synthetic handle.
    pub fn to_remote(&self, local: u64) -> Option<u64> {
        self.local_to_remote.get(&local).map(|v| *v)
    }

    /// Look up the local synthetic handle for a remote handle.
    pub fn to_local(&self, remote: u64) -> Option<u64> {
        self.remote_to_local.get(&remote).map(|v| *v)
    }

    /// Remove a mapping by local handle. Returns the remote handle if found.
    pub fn remove_by_local(&self, local: u64) -> Option<u64> {
        if let Some((_, remote)) = self.local_to_remote.remove(&local) {
            self.remote_to_local.remove(&remote);
            Some(remote)
        } else {
            None
        }
    }

    /// Number of active mappings.
    pub fn len(&self) -> usize {
        self.local_to_remote.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.local_to_remote.is_empty()
    }
}

/// Collection of all handle translation tables for a client connection.
pub struct HandleStore {
    pub contexts: HandleMap,
    pub device_ptrs: HandleMap,
    pub modules: HandleMap,
    pub functions: HandleMap,
    pub streams: HandleMap,
    pub events: HandleMap,
    pub mem_pools: HandleMap,
    pub link_states: HandleMap,
    pub libraries: HandleMap,
    pub kernels: HandleMap,
    pub graphs: HandleMap,
    pub graph_execs: HandleMap,
}

impl HandleStore {
    /// Create a new handle store with all empty tables.
    pub fn new() -> Self {
        Self {
            contexts: HandleMap::new(CONTEXT_PREFIX),
            device_ptrs: HandleMap::new(DEVICEPTR_PREFIX),
            modules: HandleMap::new(MODULE_PREFIX),
            functions: HandleMap::new(FUNCTION_PREFIX),
            streams: HandleMap::new(STREAM_PREFIX),
            events: HandleMap::new(EVENT_PREFIX),
            mem_pools: HandleMap::new(MEMPOOL_PREFIX),
            link_states: HandleMap::new(LINKSTATE_PREFIX),
            libraries: HandleMap::new(LIBRARY_PREFIX),
            kernels: HandleMap::new(KERNEL_PREFIX),
            graphs: HandleMap::new(GRAPH_PREFIX),
            graph_execs: HandleMap::new(GRAPH_EXEC_PREFIX),
        }
    }
}

impl Default for HandleStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_insert_and_lookup() {
        let map = HandleMap::new(CONTEXT_PREFIX);
        let remote = 0xDEAD_BEEF;
        let local = map.insert(remote);

        // Local handle should have the prefix
        assert_eq!(local & 0xFF00_0000_0000_0000, CONTEXT_PREFIX);

        // Bidirectional lookup
        assert_eq!(map.to_remote(local), Some(remote));
        assert_eq!(map.to_local(remote), Some(local));
    }

    #[test]
    fn test_handle_duplicate_insert() {
        let map = HandleMap::new(MODULE_PREFIX);
        let remote = 0x1234;
        let local1 = map.insert(remote);
        let local2 = map.insert(remote);

        // Same remote handle should return same local handle
        assert_eq!(local1, local2);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_handle_remove() {
        let map = HandleMap::new(DEVICEPTR_PREFIX);
        let remote = 0xAAAA;
        let local = map.insert(remote);

        assert_eq!(map.remove_by_local(local), Some(remote));
        assert_eq!(map.to_remote(local), None);
        assert_eq!(map.to_local(remote), None);
        assert!(map.is_empty());
    }

    #[test]
    fn test_handle_store() {
        let store = HandleStore::new();
        let ctx_local = store.contexts.insert(0x100);
        let ptr_local = store.device_ptrs.insert(0x200);

        // Different prefixes
        assert_ne!(ctx_local & 0xFF00_0000_0000_0000, ptr_local & 0xFF00_0000_0000_0000);
    }

    #[test]
    fn test_prefix_mask_matches_all_handle_types() {
        // All known prefixes should be detected by PREFIX_MASK
        let prefixes = [
            CONTEXT_PREFIX, DEVICEPTR_PREFIX, MODULE_PREFIX, FUNCTION_PREFIX,
            STREAM_PREFIX, EVENT_PREFIX, MEMPOOL_PREFIX, LINKSTATE_PREFIX,
            LIBRARY_PREFIX, KERNEL_PREFIX, GRAPH_PREFIX, GRAPH_EXEC_PREFIX,
        ];
        for prefix in prefixes {
            assert_ne!(prefix & PREFIX_MASK, 0, "prefix 0x{:016X} should be detected", prefix);
        }
    }

    #[test]
    fn test_prefix_mask_does_not_match_real_gpu_addresses() {
        // Real NVIDIA GPU virtual addresses are in the low range (e.g., 0x7f..., 0x0000_0003...)
        let real_addresses: Vec<u64> = vec![
            0x0000_0000_0000_0000,
            0x0000_0003_0200_0000, // typical CUDA alloc
            0x00FF_FFFF_FFFF_FFFF, // max "normal" address
            0x7F00_0000_0000_0000, // user-space high bit
        ];
        for addr in real_addresses {
            assert!(
                !is_synthetic_handle(addr),
                "real address 0x{:016X} should NOT be detected as synthetic",
                addr
            );
        }
    }

    #[test]
    fn test_is_synthetic_handle() {
        // Synthetic handles have prefix in the high byte
        let store = HandleStore::new();
        let local_ptr = store.device_ptrs.insert(0x1000);
        let local_func = store.functions.insert(0x2000);
        assert!(is_synthetic_handle(local_ptr));
        assert!(is_synthetic_handle(local_func));

        // Zero and small values are not synthetic
        assert!(!is_synthetic_handle(0));
        assert!(!is_synthetic_handle(42));
    }

    #[test]
    fn test_translate_device_ptrs_in_params() {
        let store = HandleStore::new();

        // Allocate two synthetic device pointers
        let real_a = 0x0000_0003_0200_0000u64;
        let real_b = 0x0000_0003_0300_0000u64;
        let syn_a = store.device_ptrs.insert(real_a);
        let syn_b = store.device_ptrs.insert(real_b);

        // Build a kernel params buffer: [4B num_params=2][4B size=8][8B syn_a][4B size=8][8B syn_b]
        let mut params = Vec::new();
        params.extend_from_slice(&2u32.to_le_bytes());
        params.extend_from_slice(&8u32.to_le_bytes());
        params.extend_from_slice(&syn_a.to_le_bytes());
        params.extend_from_slice(&8u32.to_le_bytes());
        params.extend_from_slice(&syn_b.to_le_bytes());

        translate_device_ptrs_in_params(&mut params, &store);

        // Verify the synthetic handles were replaced with real addresses
        let translated_a = u64::from_le_bytes(params[8..16].try_into().unwrap());
        let translated_b = u64::from_le_bytes(params[20..28].try_into().unwrap());
        assert_eq!(translated_a, real_a);
        assert_eq!(translated_b, real_b);
    }

    #[test]
    fn test_translate_device_ptrs_preserves_non_pointers() {
        let store = HandleStore::new();

        // A kernel param that is NOT a device pointer (e.g., an integer constant)
        let plain_value = 42u64;
        let mut params = Vec::new();
        params.extend_from_slice(&1u32.to_le_bytes());
        params.extend_from_slice(&8u32.to_le_bytes());
        params.extend_from_slice(&plain_value.to_le_bytes());

        translate_device_ptrs_in_params(&mut params, &store);

        // Value should be unchanged
        let result = u64::from_le_bytes(params[8..16].try_into().unwrap());
        assert_eq!(result, plain_value);
    }

    #[test]
    fn test_translate_device_ptrs_mixed_sizes() {
        let store = HandleStore::new();
        let real_ptr = 0x0000_0003_0400_0000u64;
        let syn_ptr = store.device_ptrs.insert(real_ptr);

        // Params: [4B num=2][4B size=4][4B int_val][4B size=8][8B syn_ptr]
        let int_val = 1024u32;
        let mut params = Vec::new();
        params.extend_from_slice(&2u32.to_le_bytes());
        // Param 0: 4-byte integer (not a pointer, too small for u64)
        params.extend_from_slice(&4u32.to_le_bytes());
        params.extend_from_slice(&int_val.to_le_bytes());
        // Param 1: 8-byte device pointer
        params.extend_from_slice(&8u32.to_le_bytes());
        params.extend_from_slice(&syn_ptr.to_le_bytes());

        translate_device_ptrs_in_params(&mut params, &store);

        // 4-byte param unchanged
        let int_result = u32::from_le_bytes(params[8..12].try_into().unwrap());
        assert_eq!(int_result, int_val);
        // 8-byte param translated
        let ptr_result = u64::from_le_bytes(params[16..24].try_into().unwrap());
        assert_eq!(ptr_result, real_ptr);
    }

    #[test]
    fn test_translate_device_ptrs_empty_params() {
        let store = HandleStore::new();
        let mut params = Vec::new();
        params.extend_from_slice(&0u32.to_le_bytes()); // num_params = 0

        translate_device_ptrs_in_params(&mut params, &store);

        // Should not panic, params unchanged
        assert_eq!(params.len(), 4);
        let num = u32::from_le_bytes(params[..4].try_into().unwrap());
        assert_eq!(num, 0);
    }
}
