//! Eviction policies for the memory tiering subsystem.
//!
//! Three policies are provided, each suited to a different tier:
//!
//! - **ARC** (Adaptive Replacement Cache): For VRAM tiers. Self-tuning between
//!   recency and frequency using ghost lists.
//! - **CAR** (Clock with Adaptive Replacement): For DRAM tiers. ARC-like
//!   adaptivity with O(1) clock-based victim selection.
//! - **CLOCK**: For NVMe tiers. Simple and efficient for large, slow storage.

mod arc;
mod car;
mod clock;

pub use arc::ArcPolicy;
pub use car::CarPolicy;
pub use clock::ClockPolicy;

#[cfg(test)]
mod tests;
