//! Tier status and prefetch prediction types.

use crate::memory::types::TierId;

/// A prediction that a virtual page will be needed soon and should be prefetched.
#[derive(Debug, Clone)]
pub struct PrefetchPrediction {
    /// Virtual page number predicted to be accessed
    pub vpn: u64,
    /// Current tier where the page resides
    pub source_tier: TierId,
    /// Confidence of the prediction (0.0 to 1.0)
    pub confidence: f32,
}
