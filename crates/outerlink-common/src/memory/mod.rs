//! Memory tiering subsystem types and traits.
//!
//! This module defines the 6-tier memory hierarchy (Local VRAM, Remote VRAM,
//! Local DRAM, Remote DRAM, Local NVMe, Remote NVMe) and the interfaces for
//! page table operations, tier management, eviction, migration, and access
//! monitoring.

pub mod access_monitor;
pub mod coherency;
pub mod compression;
pub mod config;
pub mod dedup;
#[cfg(test)]
mod dedup_tests;
pub mod fault_handler;
pub mod eviction;
pub mod migration;
pub mod migration_engine;
pub mod page_table;
pub mod prefetch;
pub mod pte;
pub mod tier_manager;
pub mod tier_status;
pub mod topology;
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
pub use compression::{AdaptiveCompressor, CompressionAlgorithm, CompressionConfig};
pub use coherency::{CoherencyDirectory, CoherencyStats, DirectoryEntry, PageState, ReadResponse, WriteResponse};
pub use dedup::{DedupConfig, DedupManager, DedupStats};
pub use fault_handler::{FaultConfig, FaultHandler, FaultResult, FaultStats, ThrashingConfig, ThrashingDetector, ThrashingLevel};
pub use eviction::{ArcPolicy, CarPolicy, ClockPolicy};
pub use migration_engine::TieredMigrationEngine;
pub use page_table::RobinHoodPageTable;
pub use prefetch::{PredictionSource, PrefetchConfig, PrefetchScheduler, PrefetchStats};
pub use topology::{LinkInfo, LinkType, NodeInfo, PlacementScorer, PlacementWeights, Route, TopologyGraph};
pub use tier_manager::DefaultTierManager;
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
