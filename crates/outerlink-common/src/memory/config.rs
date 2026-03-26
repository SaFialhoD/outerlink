//! Tier configuration and eviction policy definitions.
//!
//! Each of the 6 tiers has a default configuration specifying capacity,
//! bandwidth, latency, batch sizes, and the eviction policy to use.

use super::types::{TierId, TIER_COUNT, LOCAL_VRAM, REMOTE_VRAM, LOCAL_DRAM, REMOTE_DRAM, LOCAL_NVME, REMOTE_NVME};

/// Which eviction policy algorithm a tier uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionPolicyType {
    /// Adaptive Replacement Cache (balances recency and frequency).
    Arc,
    /// Clock with Adaptive Replacement (scan-resistant, low overhead).
    Car,
    /// CLOCK algorithm (simple, efficient for large working sets).
    Clock,
}

/// Configuration for a single memory tier.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Tier identifier (0-5).
    pub tier_id: TierId,
    /// Human-readable name for this tier.
    pub name: &'static str,
    /// Maximum capacity in bytes (0 = unlimited/dynamic).
    pub capacity_bytes: u64,
    /// Maximum bandwidth in bytes/second.
    pub bandwidth_bytes_per_sec: u64,
    /// Expected access latency in nanoseconds.
    pub latency_ns: u64,
    /// Number of pages to migrate in a single batch.
    pub migration_batch_size: u32,
    /// Minimum time (ms) a page must reside in this tier before eviction.
    pub min_residency_ms: u32,
    /// Eviction policy algorithm for this tier.
    pub eviction_policy: EvictionPolicyType,
    /// Whether this tier is local to the node (vs. remote/networked).
    pub is_local: bool,
    /// Whether data in this tier survives process restart (NVMe tiers).
    pub is_persistent: bool,
}

/// Returns the default tier configurations for all 6 tiers.
///
/// These defaults are based on typical hardware:
/// - VRAM: 24 GB (RTX 3090), ~900 GB/s local, ~12.5 GB/s remote (100 Gbps link)
/// - DRAM: 64 GB per node, ~50 GB/s local, ~12.5 GB/s remote
/// - NVMe: 2 TB per node, ~7 GB/s local, ~3 GB/s remote
pub fn default_tier_configs() -> [TierConfig; TIER_COUNT] {
    [
        TierConfig {
            tier_id: LOCAL_VRAM,
            name: "Local VRAM",
            capacity_bytes: 24 * 1024 * 1024 * 1024,            // 24 GB
            bandwidth_bytes_per_sec: 936 * 1024 * 1024 * 1024,  // 936 GB/s (3090)
            latency_ns: 300,                                      // ~300 ns GPU memory access
            migration_batch_size: 64,
            min_residency_ms: 100,
            eviction_policy: EvictionPolicyType::Arc,
            is_local: true,
            is_persistent: false,
        },
        TierConfig {
            tier_id: REMOTE_VRAM,
            name: "Remote VRAM",
            capacity_bytes: 24 * 1024 * 1024 * 1024,            // 24 GB
            bandwidth_bytes_per_sec: 12_500_000_000,             // 12.5 GB/s (100 Gbps)
            latency_ns: 5_000,                                    // ~5 us network + GPU
            migration_batch_size: 32,
            min_residency_ms: 500,
            eviction_policy: EvictionPolicyType::Arc,
            is_local: false,
            is_persistent: false,
        },
        TierConfig {
            tier_id: LOCAL_DRAM,
            name: "Local DRAM",
            capacity_bytes: 64 * 1024 * 1024 * 1024,            // 64 GB
            bandwidth_bytes_per_sec: 50 * 1024 * 1024 * 1024,   // 50 GB/s (DDR4 dual-channel)
            latency_ns: 100,                                      // ~100 ns DRAM
            migration_batch_size: 128,
            min_residency_ms: 50,
            eviction_policy: EvictionPolicyType::Car,
            is_local: true,
            is_persistent: false,
        },
        TierConfig {
            tier_id: REMOTE_DRAM,
            name: "Remote DRAM",
            capacity_bytes: 64 * 1024 * 1024 * 1024,            // 64 GB
            bandwidth_bytes_per_sec: 12_500_000_000,             // 12.5 GB/s
            latency_ns: 10_000,                                   // ~10 us network + DRAM
            migration_batch_size: 64,
            min_residency_ms: 1000,
            eviction_policy: EvictionPolicyType::Car,
            is_local: false,
            is_persistent: false,
        },
        TierConfig {
            tier_id: LOCAL_NVME,
            name: "Local NVMe",
            capacity_bytes: 2 * 1024 * 1024 * 1024 * 1024,     // 2 TB
            bandwidth_bytes_per_sec: 7 * 1024 * 1024 * 1024,    // 7 GB/s (Gen4 NVMe)
            latency_ns: 10_000,                                   // ~10 us NVMe
            migration_batch_size: 256,
            min_residency_ms: 5000,
            eviction_policy: EvictionPolicyType::Clock,
            is_local: true,
            is_persistent: true,
        },
        TierConfig {
            tier_id: REMOTE_NVME,
            name: "Remote NVMe",
            capacity_bytes: 2 * 1024 * 1024 * 1024 * 1024,     // 2 TB
            bandwidth_bytes_per_sec: 3 * 1024 * 1024 * 1024,    // 3 GB/s
            latency_ns: 50_000,                                   // ~50 us network + NVMe
            migration_batch_size: 128,
            min_residency_ms: 10_000,
            eviction_policy: EvictionPolicyType::Clock,
            is_local: false,
            is_persistent: true,
        },
    ]
}
