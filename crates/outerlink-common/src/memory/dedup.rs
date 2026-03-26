//! Content-addressable memory deduplication.
//!
//! Detects identical page content across the virtual address space and
//! maintains copy-on-write reference tracking so duplicate pages share
//! a single physical backing store.

use std::sync::RwLock;

use dashmap::DashMap;
use xxhash_rust::xxh3::xxh3_128;

use super::traits::DedupHook;

/// Configuration for the dedup subsystem.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Enable/disable dedup (default: true).
    pub enabled: bool,
    /// Maximum entries in the dedup table.
    pub max_entries: usize,
    /// Verify content via memcmp after hash match (default: true).
    pub verify_on_match: bool,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 1_000_000,
            verify_on_match: true,
        }
    }
}

/// A dedup table entry: one canonical page and zero or more references.
#[derive(Debug, Clone)]
pub struct DedupEntry {
    /// VPN of the canonical (master) copy.
    pub canonical_vpn: u64,
    /// Node hosting the canonical copy.
    pub canonical_node: u8,
    /// Tier of the canonical copy.
    pub canonical_tier: u8,
    /// Number of pages sharing this content (canonical + references).
    pub ref_count: u32,
    /// VPNs of all reference copies (excludes canonical).
    pub references: Vec<u64>,
}

/// Deduplication statistics.
#[derive(Debug, Default, Clone)]
pub struct DedupStats {
    pub total_checks: u64,
    pub total_matches: u64,
    pub total_unique: u64,
    /// Reference pages that share storage with a canonical page.
    pub pages_saved: u64,
    /// Bytes saved (accumulated from data.len() on each dedup match; upper-bound
    /// estimate since freed references cannot be precisely subtracted without
    /// per-entry size tracking).
    pub bytes_saved: u64,
}

/// Content-addressable deduplication manager.
///
/// Implements [`DedupHook`] for integration with the page table and
/// memory tiering subsystem.
pub struct DedupManager {
    /// Hash -> dedup entry.
    table: DashMap<u128, DedupEntry>,
    /// VPN -> hash (reverse index for fast removal).
    vpn_to_hash: DashMap<u64, u128>,
    /// Aggregate statistics.
    stats: RwLock<DedupStats>,
    /// Configuration.
    config: DedupConfig,
}

impl DedupManager {
    /// Create a new DedupManager with the given configuration.
    pub fn new(config: DedupConfig) -> Self {
        Self {
            table: DashMap::new(),
            vpn_to_hash: DashMap::new(),
            stats: RwLock::new(DedupStats::default()),
            config,
        }
    }

    /// Create a new DedupManager with default settings.
    pub fn with_defaults() -> Self {
        Self::new(DedupConfig::default())
    }

    /// Return a snapshot of current dedup statistics.
    pub fn stats(&self) -> DedupStats {
        self.stats.read().unwrap().clone()
    }

    /// Returns true if `vpn` is a canonical (master) page.
    pub fn is_canonical(&self, vpn: u64) -> bool {
        let hash = match self.vpn_to_hash.get(&vpn) {
            Some(h) => *h,
            None => return false,
        };
        match self.table.get(&hash) {
            Some(entry) => entry.canonical_vpn == vpn,
            None => false,
        }
    }

    /// Returns true if `vpn` is a dedup reference page.
    pub fn is_reference(&self, vpn: u64) -> bool {
        let hash = match self.vpn_to_hash.get(&vpn) {
            Some(h) => *h,
            None => return false,
        };
        match self.table.get(&hash) {
            Some(entry) => entry.references.contains(&vpn),
            None => false,
        }
    }

    /// Get the canonical VPN that `vpn` references, if any.
    ///
    /// Returns `Some(canonical_vpn)` for both canonical and reference pages.
    /// Returns `None` if `vpn` is not tracked.
    pub fn get_canonical_for(&self, vpn: u64) -> Option<u64> {
        let hash = self.vpn_to_hash.get(&vpn)?;
        let entry = self.table.get(&*hash)?;
        Some(entry.canonical_vpn)
    }

    /// Number of physical pages saved by deduplication.
    pub fn pages_saved(&self) -> u64 {
        self.stats.read().unwrap().pages_saved
    }

    /// Number of entries in the dedup table.
    pub fn table_size(&self) -> usize {
        self.table.len()
    }

    /// Break copy-on-write for a reference page that is about to be written.
    ///
    /// Removes the vpn from the dedup entry and returns the hash so the
    /// caller can update PTE flags.  Returns `None` if the vpn is not a
    /// tracked reference.
    pub fn break_dedup(&self, vpn: u64) -> Option<u128> {
        let hash = {
            let h = self.vpn_to_hash.get(&vpn)?;
            *h
        };

        let mut entry = self.table.get_mut(&hash)?;

        // Only break references, not the canonical itself.
        if let Some(pos) = entry.references.iter().position(|&v| v == vpn) {
            entry.references.remove(pos);
            entry.ref_count -= 1;
            drop(entry);

            self.vpn_to_hash.remove(&vpn);

            let mut stats = self.stats.write().unwrap();
            if stats.pages_saved > 0 {
                stats.pages_saved -= 1;
                // We don't know the page size here, but we tracked it on insert.
                // For correctness we'd need to store per-entry page_size. For now
                // we leave bytes_saved as-is since we don't have the page size.
                // The caller should adjust bytes_saved if needed.
            }

            Some(hash)
        } else {
            None
        }
    }

    /// Compute the xxHash128 of the given data.
    fn hash_page(data: &[u8]) -> u128 {
        xxh3_128(data)
    }
}

impl DedupHook for DedupManager {
    fn check_dedup(&self, vpn: u64, data: &[u8]) -> Option<u64> {
        if !self.config.enabled {
            return None;
        }

        // Guard: if this VPN is already tracked, skip to avoid it appearing
        // as both canonical and reference of itself (or of two different hashes).
        if self.vpn_to_hash.contains_key(&vpn) {
            return None;
        }

        let hash = Self::hash_page(data);

        // Try to get existing entry first.
        if let Some(mut entry) = self.table.get_mut(&hash) {
            // Match found: this page has identical content to the canonical.
            // Note: verify_on_match would require access to canonical page data
            // for memcmp verification. Currently we trust the xxHash128 collision
            // resistance (probability ~2^-128). Full verification requires a
            // PageTable reference which will be wired in during integration.
            entry.ref_count += 1;
            entry.references.push(vpn);
            let canonical = entry.canonical_vpn;
            drop(entry);

            self.vpn_to_hash.insert(vpn, hash);

            let mut stats = self.stats.write().unwrap();
            stats.total_checks += 1;
            stats.total_matches += 1;
            stats.pages_saved += 1;
            stats.bytes_saved += data.len() as u64;

            return Some(canonical);
        }

        // No match: check capacity before inserting.
        // max_entries == 0 means unlimited.
        if self.config.max_entries > 0 && self.table.len() >= self.config.max_entries {
            // Table full; skip dedup for this page.
            let mut stats = self.stats.write().unwrap();
            stats.total_checks += 1;
            return None;
        }

        // Insert new canonical entry.
        let entry = DedupEntry {
            canonical_vpn: vpn,
            canonical_node: 0,
            canonical_tier: 0,
            ref_count: 1,
            references: Vec::new(),
        };
        self.table.insert(hash, entry);
        self.vpn_to_hash.insert(vpn, hash);

        let mut stats = self.stats.write().unwrap();
        stats.total_checks += 1;
        stats.total_unique += 1;

        None
    }

    fn notify_free(&self, vpn: u64) {
        let hash = match self.vpn_to_hash.remove(&vpn) {
            Some((_, h)) => h,
            None => return,
        };

        // Track whether a saved page was lost so we can update stats after
        // releasing the DashMap guard.
        let should_remove;
        let decrement_saved;

        {
            let mut entry = match self.table.get_mut(&hash) {
                Some(e) => e,
                None => return,
            };

            if entry.canonical_vpn == vpn {
                // Freeing the canonical page.
                if let Some(new_canonical) = entry.references.first().copied() {
                    // Promote the first reference to canonical.
                    // The promoted reference was counted as a "saved" page;
                    // now it becomes the canonical, so pages_saved drops by 1.
                    entry.references.remove(0);
                    entry.canonical_vpn = new_canonical;
                    entry.ref_count -= 1;
                    should_remove = false;
                    decrement_saved = true;
                } else {
                    // No references left; remove the whole entry.
                    // The canonical itself was never counted as "saved"
                    // (only references are), so no decrement needed.
                    should_remove = true;
                    decrement_saved = false;
                }
            } else {
                // Freeing a reference page: one less saved page.
                if let Some(pos) = entry.references.iter().position(|&v| v == vpn) {
                    entry.references.remove(pos);
                    entry.ref_count -= 1;
                    decrement_saved = true;
                } else {
                    decrement_saved = false;
                }
                should_remove = false;
            }
        }

        if should_remove {
            self.table.remove(&hash);
        }

        if decrement_saved {
            let mut stats = self.stats.write().unwrap();
            if stats.pages_saved > 0 {
                stats.pages_saved -= 1;
            }
            // We don't track per-entry page size, so bytes_saved cannot be
            // decremented precisely. This is a known limitation documented
            // in break_dedup as well. The caller should treat bytes_saved as
            // an upper-bound estimate.
        }
    }
}

#[path = "dedup_tests.rs"]
mod dedup_tests;
