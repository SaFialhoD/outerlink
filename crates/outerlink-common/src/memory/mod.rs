//! Memory management subsystem for OuterLink.
//!
//! Provides coherency directory (distributed page ownership tracking),
//! fault handling (pre-launch page resolution), and thrashing detection.

pub mod coherency;
pub mod fault_handler;

#[cfg(test)]
mod tests;
