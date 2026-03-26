//! Memory tiering subsystem for OuterLink.
//!
//! Provides types, traits, and eviction policies for managing pages across
//! a multi-tier memory hierarchy (VRAM, DRAM, NVMe).

pub mod eviction;
pub mod tier_status;
pub mod traits;
pub mod types;

pub use eviction::{ArcPolicy, CarPolicy, ClockPolicy};
pub use tier_status::EvictionCandidate;
pub use traits::EvictionPolicy;
pub use types::{AccessPatternType, TierId, DRAM_TIERS, NVME_TIERS, VRAM_TIERS};
