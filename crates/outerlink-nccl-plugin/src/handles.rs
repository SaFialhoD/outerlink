//! Handle management for the NCCL plugin.
//!
//! NCCL passes opaque `*mut c_void` pointers across the FFI boundary. We map
//! these to `u64` IDs stored in a concurrent `DashMap`, so Rust owns all the
//! actual state and the C side only ever holds integer handles encoded as pointers.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// A thread-safe table mapping opaque u64 IDs to Rust objects of type `T`.
pub struct HandleTable<T> {
    map: DashMap<u64, T>,
    next_id: AtomicU64,
}

impl<T> HandleTable<T> {
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
            // Start at 1 so that 0 / null pointer is never a valid handle.
            next_id: AtomicU64::new(1),
        }
    }

    /// Insert a value and return its unique handle ID.
    pub fn insert(&self, value: T) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.map.insert(id, value);
        id
    }

    /// Get a shared reference to the value behind `id`.
    pub fn get(&self, id: u64) -> Option<dashmap::mapref::one::Ref<'_, u64, T>> {
        self.map.get(&id)
    }

    /// Get a mutable reference to the value behind `id`.
    pub fn get_mut(&self, id: u64) -> Option<dashmap::mapref::one::RefMut<'_, u64, T>> {
        self.map.get_mut(&id)
    }

    /// Remove and return the value behind `id`.
    pub fn remove(&self, id: u64) -> Option<T> {
        self.map.remove(&id).map(|(_, v)| v)
    }

    /// Returns the number of entries in the table.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }
}
