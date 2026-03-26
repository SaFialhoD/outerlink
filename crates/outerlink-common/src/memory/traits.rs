//! Memory subsystem traits.
//!
//! These traits define the interfaces for memory operations including
//! compression hooks used by the transport layer.

/// Hook for transparent compression/decompression of data moving between tiers.
///
/// Implementations decide whether data is worth compressing based on size,
/// entropy, and expected compression ratio. The transport layer calls
/// `try_compress` before sending and `decompress` on receipt.
pub trait CompressionHook: Send + Sync {
    /// Attempt to compress the given data.
    ///
    /// Returns `Some((compressed_data, ratio))` if compression is worthwhile,
    /// where ratio = original_size / compressed_size (always >= 1.0).
    /// Returns `None` if compression should be skipped (data too small,
    /// incompressible, or ratio below threshold).
    fn try_compress(&self, data: &[u8]) -> Option<(Vec<u8>, f32)>;

    /// Decompress data that was previously compressed.
    ///
    /// `original_size` is the size of the data before compression,
    /// used to pre-allocate the output buffer.
    fn decompress(&self, compressed: &[u8], original_size: usize) -> Vec<u8>;
}
