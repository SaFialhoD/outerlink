//! Memory tiering subsystem types and traits.
//!
//! This module defines the 6-tier memory hierarchy (Local VRAM, Remote VRAM,
//! Local DRAM, Remote DRAM, Local NVMe, Remote NVMe) and the interfaces for
//! page table operations, tier management, eviction, migration, and access
//! monitoring.

pub mod access_monitor;
pub mod config;
pub mod eviction;
pub mod migration;
pub mod migration_engine;
pub mod page_table;
pub mod pte;
pub mod tier_status;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export the most commonly used types at the module level.
pub use config::{default_tier_configs, EvictionPolicyType, TierConfig};
pub use migration::{
    MigrationError, MigrationHandle, MigrationPriority, MigrationReason, MigrationRequest,
    MigrationStatus,
};
pub use access_monitor::{InterceptionAccessMonitor, MonitorConfig};
pub use eviction::{ArcPolicy, CarPolicy, ClockPolicy};
pub use migration_engine::TieredMigrationEngine;
pub use page_table::RobinHoodPageTable;
pub use pte::{PageTableEntry, PteFlags};
pub use tier_status::{
    AllocationError, AllocationHints, EvictionCandidate, PrefetchPrediction, TierAllocation,
    TierStatus,
};
pub use traits::{
    AccessMonitor, AccessType, CompressionHook, DedupHook, EvictionPolicy, MigrationEngine,
    PageTable, ParityHook, PinError, TierManager,
};
pub use types::{
    AccessPatternType, CoherencyField, CoherencyState, MemcpyDirection, NodeId, PressureLevel,
    TierId, LOCAL_DRAM, LOCAL_NVME, LOCAL_VRAM, REMOTE_DRAM, REMOTE_NVME, REMOTE_VRAM,
    TIER_COUNT,
};
