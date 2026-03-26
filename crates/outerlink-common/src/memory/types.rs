//! Memory tier types shared across the memory subsystem.

/// Identifies a memory tier in the system.
///
/// Tiers represent different memory backends (VRAM, host RAM, remote VRAM, etc.)
/// ordered by performance characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TierId(pub u8);

impl TierId {
    /// GPU VRAM (fastest, most constrained).
    pub const VRAM: TierId = TierId(0);
    /// Pinned host memory (CPU RAM, DMA-accessible).
    pub const HOST_PINNED: TierId = TierId(1);
    /// Pageable host memory.
    pub const HOST_PAGEABLE: TierId = TierId(2);
    /// Remote GPU VRAM (accessed over network).
    pub const REMOTE_VRAM: TierId = TierId(3);
}

impl std::fmt::Display for TierId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            0 => write!(f, "VRAM"),
            1 => write!(f, "HostPinned"),
            2 => write!(f, "HostPageable"),
            3 => write!(f, "RemoteVRAM"),
            n => write!(f, "Tier({})", n),
        }
    }
}
