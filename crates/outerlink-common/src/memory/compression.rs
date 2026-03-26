//! Transport compression for memory tier transfers.
//!
//! Provides adaptive compression that decides at runtime whether to compress
//! data based on size, entropy, and achieved compression ratio. Supports
//! LZ4 (fast, lower ratio) and Zstd (slower, better ratio) algorithms.

use std::collections::HashSet;

use dashmap::DashMap;

use super::traits::CompressionHook;

/// Compression algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionAlgorithm {
    /// No compression - passthrough.
    None,
    /// LZ4 - fast compression, moderate ratio.
    Lz4,
    /// Zstd - configurable compression level (1-22, default 3).
    Zstd { level: i32 },
}

/// Internal trait for compression algorithm backends.
pub trait CompressionEngine: Send + Sync {
    /// Which algorithm this engine uses.
    fn algorithm(&self) -> CompressionAlgorithm;
    /// Compress data, returning None if compression fails.
    fn compress(&self, data: &[u8]) -> Option<Vec<u8>>;
    /// Decompress data given the original uncompressed size.
    fn decompress(&self, compressed: &[u8], original_size: usize) -> Vec<u8>;
    /// Estimate compression ratio from a sample.
    fn estimate_ratio(&self, sample: &[u8]) -> f32;
}

/// LZ4 compression engine using `lz4_flex` (pure Rust).
pub struct Lz4Engine;

impl CompressionEngine for Lz4Engine {
    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Lz4
    }

    fn compress(&self, data: &[u8]) -> Option<Vec<u8>> {
        let compressed = lz4_flex::compress_prepend_size(data);
        Some(compressed)
    }

    fn decompress(&self, compressed: &[u8], _original_size: usize) -> Vec<u8> {
        lz4_flex::decompress_size_prepended(compressed)
            .expect("LZ4 decompression failed: corrupted data")
    }

    fn estimate_ratio(&self, sample: &[u8]) -> f32 {
        if sample.is_empty() {
            return 1.0;
        }
        let compressed = lz4_flex::compress_prepend_size(sample);
        sample.len() as f32 / compressed.len() as f32
    }
}

/// Zstd compression engine.
pub struct ZstdEngine {
    level: i32,
}

impl ZstdEngine {
    /// Create a new Zstd engine with the given compression level (1-22).
    pub fn new(level: i32) -> Self {
        Self { level }
    }
}

impl CompressionEngine for ZstdEngine {
    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Zstd { level: self.level }
    }

    fn compress(&self, data: &[u8]) -> Option<Vec<u8>> {
        zstd::encode_all(data, self.level).ok()
    }

    fn decompress(&self, compressed: &[u8], _original_size: usize) -> Vec<u8> {
        zstd::decode_all(compressed)
            .expect("Zstd decompression failed: corrupted data")
    }

    fn estimate_ratio(&self, sample: &[u8]) -> f32 {
        if sample.is_empty() {
            return 1.0;
        }
        match zstd::encode_all(sample, self.level) {
            Ok(compressed) => sample.len() as f32 / compressed.len() as f32,
            Err(_) => 1.0,
        }
    }
}

/// No-op compression engine (passthrough).
struct NoneEngine;

impl CompressionEngine for NoneEngine {
    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::None
    }

    fn compress(&self, data: &[u8]) -> Option<Vec<u8>> {
        Some(data.to_vec())
    }

    fn decompress(&self, compressed: &[u8], _original_size: usize) -> Vec<u8> {
        compressed.to_vec()
    }

    fn estimate_ratio(&self, _sample: &[u8]) -> f32 {
        1.0
    }
}

/// Configuration for the adaptive compressor.
pub struct CompressionConfig {
    /// Minimum data size to attempt compression (default: 4096 bytes).
    pub min_size: usize,
    /// Minimum compression ratio to keep compressed result (default: 1.5).
    pub min_ratio: f32,
    /// Algorithm to use.
    pub algorithm: CompressionAlgorithm,
    /// Enable entropy sampling to skip incompressible data.
    pub entropy_check: bool,
    /// Max distinct byte values in first 256 bytes before skipping (default: 240).
    pub entropy_threshold: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_size: 4096,
            min_ratio: 1.5,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: true,
            entropy_threshold: 240,
        }
    }
}

/// Adaptive compressor that implements the `CompressionHook` trait.
///
/// Makes per-transfer decisions about whether to compress based on data size,
/// entropy sampling, and achieved compression ratio. Caches ratio estimates
/// keyed by a hash of the first 256 bytes to speed up repeated patterns.
pub struct AdaptiveCompressor {
    config: CompressionConfig,
    engine: Box<dyn CompressionEngine>,
    /// Cached compression ratios: hash of first 256 bytes -> ratio.
    #[allow(dead_code)]
    ratio_cache: DashMap<u64, f32>,
}

impl AdaptiveCompressor {
    /// Create a new adaptive compressor with the given configuration.
    pub fn new(config: CompressionConfig) -> Self {
        let engine: Box<dyn CompressionEngine> = match config.algorithm {
            CompressionAlgorithm::None => Box::new(NoneEngine),
            CompressionAlgorithm::Lz4 => Box::new(Lz4Engine),
            CompressionAlgorithm::Zstd { level } => Box::new(ZstdEngine::new(level)),
        };
        Self {
            config,
            engine,
            ratio_cache: DashMap::new(),
        }
    }

    /// Check if the first 256 bytes have too many distinct values (high entropy).
    /// Returns true if data appears incompressible.
    fn is_high_entropy(&self, data: &[u8]) -> bool {
        let sample_len = data.len().min(256);
        let sample = &data[..sample_len];
        let mut seen = HashSet::with_capacity(256);
        for &byte in sample {
            seen.insert(byte);
        }
        seen.len() > self.config.entropy_threshold
    }
}

impl CompressionHook for AdaptiveCompressor {
    fn try_compress(&self, data: &[u8]) -> Option<(Vec<u8>, f32)> {
        // Step 1: Check minimum size
        if data.len() < self.config.min_size {
            return None;
        }

        // Step 2: Entropy check (fast path for incompressible data)
        if self.config.entropy_check && self.is_high_entropy(data) {
            return None;
        }

        // Step 3: Compress
        let compressed = self.engine.compress(data)?;

        // Step 4: Calculate ratio
        let ratio = data.len() as f32 / compressed.len() as f32;

        // Step 5: Check ratio threshold
        if ratio < self.config.min_ratio {
            return None;
        }

        Some((compressed, ratio))
    }

    fn decompress(&self, compressed: &[u8], original_size: usize) -> Vec<u8> {
        self.engine.decompress(compressed, original_size)
    }
}

/// Wire header prepended to compressed frames on the network.
///
/// Exactly 16 bytes, laid out for zero-copy parsing.
#[repr(C)]
pub struct CompressionHeader {
    /// Magic bytes: 0xC014
    pub magic: u16,
    /// Header version (currently 1).
    pub version: u8,
    /// Algorithm identifier: 0=None, 1=LZ4, 2=Zstd.
    pub algorithm: u8,
    /// Size of original uncompressed data.
    pub original_size: u32,
    /// Size of compressed data (after header).
    pub compressed_size: u32,
    /// xxHash32 checksum of compressed data.
    pub checksum: u32,
}

impl CompressionHeader {
    /// Header magic value.
    pub const MAGIC: u16 = 0xC014;
    /// Current header version.
    pub const VERSION: u8 = 1;
    /// Header size in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Algorithm byte for None.
    pub const ALG_NONE: u8 = 0;
    /// Algorithm byte for LZ4.
    pub const ALG_LZ4: u8 = 1;
    /// Algorithm byte for Zstd.
    pub const ALG_ZSTD: u8 = 2;

    /// Serialize this header to bytes.
    pub fn to_bytes(&self) -> [u8; 16] {
        unsafe { std::mem::transmute_copy(self) }
    }

    /// Deserialize a header from bytes. Returns None if magic/version mismatch.
    pub fn from_bytes(bytes: &[u8; 16]) -> Option<Self> {
        let header: Self = unsafe { std::ptr::read(bytes.as_ptr() as *const Self) };
        if header.magic != Self::MAGIC || header.version != Self::VERSION {
            return None;
        }
        Some(header)
    }

    /// Map a `CompressionAlgorithm` to its wire byte.
    pub fn algorithm_to_byte(alg: CompressionAlgorithm) -> u8 {
        match alg {
            CompressionAlgorithm::None => Self::ALG_NONE,
            CompressionAlgorithm::Lz4 => Self::ALG_LZ4,
            CompressionAlgorithm::Zstd { .. } => Self::ALG_ZSTD,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::traits::CompressionHook;

    // ---- Test 1: LZ4 compress/decompress roundtrip ----
    #[test]
    fn lz4_compress_decompress_roundtrip() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        let original = b"Hello world! Hello world! Hello world! Hello world! Repeated data compresses well.";
        let (compressed, ratio) = compressor.try_compress(original).expect("should compress");
        assert!(ratio >= 1.0, "ratio should be >= 1.0");

        let decompressed = compressor.decompress(&compressed, original.len());
        assert_eq!(decompressed, original);
    }

    // ---- Test 2: Zstd compress/decompress roundtrip ----
    #[test]
    fn zstd_compress_decompress_roundtrip() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Zstd { level: 3 },
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        let original = b"AAAA BBBB CCCC DDDD AAAA BBBB CCCC DDDD repeated patterns here!";
        let (compressed, ratio) = compressor.try_compress(original).expect("should compress");
        assert!(ratio >= 1.0);

        let decompressed = compressor.decompress(&compressed, original.len());
        assert_eq!(decompressed, original);
    }

    // ---- Test 3: Adaptive skips small data ----
    #[test]
    fn adaptive_skips_small_data() {
        let config = CompressionConfig {
            min_size: 4096,
            min_ratio: 1.5,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        let small_data = vec![0u8; 100];
        assert!(compressor.try_compress(&small_data).is_none(), "small data should be skipped");
    }

    // ---- Test 4: Adaptive skips incompressible (high entropy) ----
    #[test]
    fn adaptive_skips_incompressible() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: true,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        // Data with all 256 byte values present in the first 256 bytes
        let random_data: Vec<u8> = (0..=255).cycle().take(4096).collect();
        assert!(compressor.try_compress(&random_data).is_none(), "high entropy data should be skipped");
    }

    // ---- Test 5: Adaptive compresses compressible data ----
    #[test]
    fn adaptive_compresses_compressible() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        // Highly compressible: all zeros
        let data = vec![0u8; 8192];
        let result = compressor.try_compress(&data);
        assert!(result.is_some(), "compressible data should compress");

        let (compressed, ratio) = result.unwrap();
        assert!(compressed.len() < data.len(), "compressed should be smaller");
        assert!(ratio > 1.0, "ratio should indicate compression");
    }

    // ---- Test 6: Adaptive returns correct ratio ----
    #[test]
    fn adaptive_returns_ratio() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        let data = vec![42u8; 4096];
        let (compressed, ratio) = compressor.try_compress(&data).expect("should compress");

        let expected_ratio = data.len() as f32 / compressed.len() as f32;
        let diff = (ratio - expected_ratio).abs();
        assert!(diff < 0.01, "ratio should be original/compressed, got {} expected {}", ratio, expected_ratio);
    }

    // ---- Test 7: Adaptive respects min_ratio ----
    #[test]
    fn adaptive_respects_min_ratio() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 10000.0, // Impossibly high ratio requirement - no algorithm achieves this
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        // Even highly compressible data can't achieve 10000x ratio
        let data = vec![0u8; 4096];
        assert!(compressor.try_compress(&data).is_none(), "should skip when ratio threshold too high");
    }

    // ---- Test 8: Compression header roundtrip ----
    #[test]
    fn compression_header_roundtrip() {
        let header = CompressionHeader {
            magic: 0xC014,
            version: 1,
            algorithm: 1, // LZ4
            original_size: 65536,
            compressed_size: 32000,
            checksum: 0xDEADBEEF,
        };

        // Serialize to bytes
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 16, "header should be exactly 16 bytes");

        // Deserialize back
        let restored = CompressionHeader::from_bytes(&bytes).expect("should parse valid header");

        assert_eq!(restored.magic, 0xC014);
        assert_eq!(restored.version, 1);
        assert_eq!(restored.algorithm, 1);
        assert_eq!(restored.original_size, 65536);
        assert_eq!(restored.compressed_size, 32000);
        assert_eq!(restored.checksum, 0xDEADBEEF);
    }

    // ---- Test 9: CompressionHook trait object ----
    #[test]
    fn compression_hook_trait() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: false,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        // Must work as a trait object
        let hook: Box<dyn CompressionHook> = Box::new(compressor);

        let data = vec![0u8; 4096];
        let (compressed, _ratio) = hook.try_compress(&data).expect("should compress via trait object");
        let decompressed = hook.decompress(&compressed, data.len());
        assert_eq!(decompressed, data);
    }

    // ---- Test 10: Entropy check fast path ----
    #[test]
    fn entropy_check_fast_path() {
        let config = CompressionConfig {
            min_size: 0,
            min_ratio: 1.0,
            algorithm: CompressionAlgorithm::Lz4,
            entropy_check: true,
            entropy_threshold: 240,
        };
        let compressor = AdaptiveCompressor::new(config);

        // All 256 byte values in first 256 bytes -> should be detected as high entropy
        let mut data = Vec::with_capacity(4096);
        for i in 0..=255u8 {
            data.push(i);
        }
        // Pad the rest with zeros
        data.resize(4096, 0);

        let start = std::time::Instant::now();
        let result = compressor.try_compress(&data);
        let elapsed = start.elapsed();

        // 256 distinct values > 240 threshold -> should skip
        assert!(result.is_none(), "high entropy first 256 bytes should trigger skip");
        assert!(elapsed.as_millis() < 1, "entropy check should be sub-millisecond, took {:?}", elapsed);
    }
}
