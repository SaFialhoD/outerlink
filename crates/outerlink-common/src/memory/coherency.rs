//! Distributed coherency directory for tracking page ownership and sharing state.
//!
//! Implements a directory-based coherency protocol where each page can be in one of
//! three states: Invalid, Shared, or Exclusive. The directory tracks which nodes
//! hold copies and coordinates transitions between states.

use dashmap::DashMap;
use std::sync::RwLock;

/// Distributed directory tracking page ownership and sharing state.
pub struct CoherencyDirectory {
    /// Per-page directory entries: vpn -> DirectoryEntry
    entries: DashMap<u64, DirectoryEntry>,
    /// This node's ID
    local_node: u8,
    stats: RwLock<CoherencyStats>,
}

/// A single page's directory entry tracking its coherency state.
pub struct DirectoryEntry {
    pub vpn: u64,
    pub home_node: u8,
    pub state: PageState,
    pub owner: Option<u8>,
    pub sharers: Vec<u8>,
}

/// The coherency state of a page in the directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    Invalid,
    Shared,
    Exclusive,
}

/// Counters for coherency protocol operations.
#[derive(Debug, Default, Clone)]
pub struct CoherencyStats {
    pub read_requests: u64,
    pub write_requests: u64,
    pub invalidations_sent: u64,
    pub upgrades: u64,
    pub downgrades: u64,
}

/// Response to a read request.
pub enum ReadResponse {
    /// Read access granted; fetch data from this node.
    Granted { data_source: u8 },
    /// Requesting node already has shared access.
    AlreadyShared,
}

/// Response to a write request.
pub enum WriteResponse {
    /// Exclusive access granted; these nodes were invalidated.
    Granted { invalidated: Vec<u8> },
    /// Requesting node already has exclusive access.
    AlreadyExclusive,
}

impl CoherencyDirectory {
    /// Create a new coherency directory for the given local node.
    pub fn new(local_node: u8) -> Self {
        Self {
            entries: DashMap::new(),
            local_node,
            stats: RwLock::new(CoherencyStats::default()),
        }
    }

    /// Handle a read request from a remote node.
    ///
    /// State transitions:
    /// - Invalid -> Shared (fetch from home node)
    /// - Shared -> Shared (add requester to sharers)
    /// - Exclusive -> Shared (downgrade owner, add requester)
    pub fn handle_read_request(&self, vpn: u64, requesting_node: u8) -> ReadResponse {
        // Perform all DashMap mutations first, collecting stat deltas.
        // Stats are updated after releasing the DashMap shard lock to avoid
        // nested lock acquisition (DashMap shard + RwLock<Stats>).
        let mut stat_downgrade = false;

        let response = {
            let mut entry = self.entries.entry(vpn).or_insert_with(|| DirectoryEntry {
                vpn,
                home_node: self.local_node,
                state: PageState::Invalid,
                owner: None,
                sharers: Vec::new(),
            });

            match entry.state {
                PageState::Invalid => {
                    entry.state = PageState::Shared;
                    entry.sharers.push(requesting_node);
                    ReadResponse::Granted {
                        data_source: entry.home_node,
                    }
                }
                PageState::Shared => {
                    if entry.sharers.contains(&requesting_node) {
                        ReadResponse::AlreadyShared
                    } else {
                        let data_source = entry.home_node;
                        entry.sharers.push(requesting_node);
                        ReadResponse::Granted { data_source }
                    }
                }
                PageState::Exclusive => {
                    let previous_owner =
                        entry.owner.expect("Exclusive state must have an owner");
                    stat_downgrade = true;

                    let data_source = previous_owner;
                    entry.state = PageState::Shared;
                    entry.owner = None;
                    entry.sharers.clear();
                    entry.sharers.push(previous_owner);
                    entry.sharers.push(requesting_node);

                    ReadResponse::Granted { data_source }
                }
            }
        }; // DashMap shard lock released here

        // Update stats after releasing DashMap guard.
        {
            let mut stats = self.stats.write().unwrap();
            stats.read_requests += 1;
            if stat_downgrade {
                stats.downgrades += 1;
            }
        }

        response
    }

    /// Handle a write request from a remote node.
    ///
    /// State transitions:
    /// - Invalid -> Exclusive (grant to requester)
    /// - Shared -> Exclusive (invalidate all sharers, grant to requester)
    /// - Exclusive (same node) -> AlreadyExclusive
    /// - Exclusive (other node) -> Exclusive (invalidate old owner, caller
    ///   must fetch current data from the invalidated node before granting)
    pub fn handle_write_request(&self, vpn: u64, requesting_node: u8) -> WriteResponse {
        // Collect stat deltas while holding the DashMap guard, then apply
        // after releasing it to avoid nested locks.
        let mut stat_invalidations: u64 = 0;
        let mut stat_upgrade = false;

        let response = {
            let mut entry = self.entries.entry(vpn).or_insert_with(|| DirectoryEntry {
                vpn,
                home_node: self.local_node,
                state: PageState::Invalid,
                owner: None,
                sharers: Vec::new(),
            });

            match entry.state {
                PageState::Invalid => {
                    entry.state = PageState::Exclusive;
                    entry.owner = Some(requesting_node);
                    entry.sharers.clear();
                    WriteResponse::Granted {
                        invalidated: Vec::new(),
                    }
                }
                PageState::Shared => {
                    let invalidated: Vec<u8> = entry
                        .sharers
                        .iter()
                        .copied()
                        .filter(|&n| n != requesting_node)
                        .collect();

                    stat_invalidations = invalidated.len() as u64;
                    stat_upgrade = entry.sharers.contains(&requesting_node);

                    entry.state = PageState::Exclusive;
                    entry.owner = Some(requesting_node);
                    entry.sharers.clear();

                    WriteResponse::Granted { invalidated }
                }
                PageState::Exclusive => {
                    let current_owner =
                        entry.owner.expect("Exclusive state must have an owner");
                    if current_owner == requesting_node {
                        WriteResponse::AlreadyExclusive
                    } else {
                        // Ownership transfer: the caller (transport layer) must
                        // fetch current page data from current_owner before
                        // granting write access to requesting_node.
                        stat_invalidations = 1;
                        let invalidated = vec![current_owner];
                        entry.owner = Some(requesting_node);

                        WriteResponse::Granted { invalidated }
                    }
                }
            }
        }; // DashMap shard lock released here

        // Update stats after releasing DashMap guard.
        {
            let mut stats = self.stats.write().unwrap();
            stats.write_requests += 1;
            stats.invalidations_sent += stat_invalidations;
            if stat_upgrade {
                stats.upgrades += 1;
            }
        }

        response
    }

    /// Handle notification that a node is evicting a page.
    ///
    /// Removes the node from sharers or clears exclusive ownership.
    /// If no nodes remain, the page transitions to Invalid.
    pub fn handle_evict_notify(&self, vpn: u64, node: u8) {
        if let Some(mut entry) = self.entries.get_mut(&vpn) {
            match entry.state {
                PageState::Exclusive => {
                    if entry.owner == Some(node) {
                        entry.state = PageState::Invalid;
                        entry.owner = None;
                    }
                }
                PageState::Shared => {
                    entry.sharers.retain(|&n| n != node);
                    if entry.sharers.is_empty() {
                        entry.state = PageState::Invalid;
                    }
                }
                PageState::Invalid => {}
            }
        }
    }

    /// Get the current coherency state of a page.
    pub fn get_state(&self, vpn: u64) -> Option<PageState> {
        self.entries.get(&vpn).map(|e| e.state)
    }

    /// Get the list of nodes currently sharing a page.
    pub fn get_sharers(&self, vpn: u64) -> Vec<u8> {
        self.entries
            .get(&vpn)
            .map(|e| e.sharers.clone())
            .unwrap_or_default()
    }

    /// Get a snapshot of the coherency statistics.
    pub fn stats(&self) -> CoherencyStats {
        self.stats.read().unwrap().clone()
    }
}
