//! Handle translation between local (synthetic) and remote (real) CUDA handles.
//!
//! The client generates synthetic handles for the application and maintains
//! bidirectional mappings to the real handles on the remote server.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Prefix ranges for synthetic handle types (high byte).
/// Makes handles debuggable - you can tell the type from the value.
const CONTEXT_PREFIX: u64 = 0x0C00_0000_0000_0000;
const DEVICEPTR_PREFIX: u64 = 0x0D00_0000_0000_0000;
const MODULE_PREFIX: u64 = 0x0E00_0000_0000_0000;
const FUNCTION_PREFIX: u64 = 0x0F00_0000_0000_0000;
const STREAM_PREFIX: u64 = 0x1000_0000_0000_0000;
const EVENT_PREFIX: u64 = 0x1100_0000_0000_0000;
const MEMPOOL_PREFIX: u64 = 0x1200_0000_0000_0000;

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
}
