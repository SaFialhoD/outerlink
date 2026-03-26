//! Memory tiering subsystem.
//!
//! Provides types and components for OuterLink's multi-tier memory management:
//! virtual page tracking, access monitoring, migration scheduling, and
//! speculative prefetching.

pub mod migration;
pub mod prefetch;
pub mod tier_status;
pub mod types;

pub use migration::{MigrationPriority, MigrationRequest};
pub use prefetch::{PredictionSource, PrefetchConfig, PrefetchScheduler, PrefetchStats};
pub use tier_status::PrefetchPrediction;
pub use types::{AccessPatternType, TierId};
