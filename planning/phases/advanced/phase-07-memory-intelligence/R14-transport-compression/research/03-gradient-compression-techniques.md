# R14 Research: Gradient Compression Techniques for Distributed ML

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Evaluate ML-specific gradient compression techniques for integration into OuterLink's transport layer. These techniques exploit the mathematical properties of gradient tensors (sparsity, low rank, quantizability) to achieve compression ratios far beyond what general-purpose algorithms can deliver (10-100x vs 2-3x). This is the highest-value compression target for OuterLink because distributed training is bandwidth-hungry and gradients are highly compressible.

## Why Gradient Compression Is Different

General-purpose compression (LZ4, Zstd) treats data as opaque byte streams. Gradient compression exploits domain knowledge:

- **Sparsity:** Most gradient values are near zero after each step. Only a small fraction carry significant information.
- **Low rank:** Gradient matrices often have effective rank much lower than their dimensions.
- **Quantizability:** Gradient values don't need full FP32 precision; 1-8 bits often suffice.
- **Temporal correlation:** Consecutive gradient steps are similar (delta encoding opportunity).
- **Error tolerance:** ML training is inherently noise-tolerant. Lossy compression converges to the same result.

These properties enable **lossy compression with no accuracy loss** -- something impossible with general data.

## Technique Taxonomy

### 1. Sparsification

**Core idea:** Only send the largest gradient values; zero out the rest.

#### Top-K Sparsification

- Select the K elements with largest absolute value
- Transmit only those K values + their indices
- Compression ratio: N/K (e.g., send top 1% = 100x compression)
- Requires sorting (O(N log N)) or approximate selection (O(N))

**Performance:**
- 50x compression achievable with minimal accuracy impact (per 2025 studies)
- Global Top-K (across all workers) gives best convergence but requires coordination
- Local Top-K (per worker) is simpler but less efficient
- Sorting overhead is significant -- DGC (Deep Gradient Compression) uses threshold-based selection instead

**Recent advances (2025):** Column-wise sparsification maintains structural integrity better than element-wise Top-K, achieving up to 90% compression rate with ~2x training throughput gain and ~12x inference throughput gain.

#### Random Sparsification

- Randomly select K elements to send (with appropriate scaling)
- Cheaper than Top-K (no sorting)
- Slightly worse convergence per iteration
- Works well with error feedback

#### Deep Gradient Compression (DGC)

- Momentum correction + local gradient clipping + momentum factor masking + warm-up training
- Achieves 270-600x compression on CNNs with no accuracy loss
- More complex to implement than basic Top-K

### 2. Quantization

**Core idea:** Reduce bit-width of gradient values.

#### 1-bit SGD / TernGrad

- Quantize each gradient to {-1, +1} (1-bit) or {-1, 0, +1} (ternary)
- 32x compression for 1-bit from FP32
- Simple and fast
- Convergence can suffer without error feedback

#### QSGD (Quantized SGD)

- Stochastic quantization to s levels
- Tunable bits: log2(s) bits per value
- Unbiased estimator preserves convergence guarantees

#### DeepSpeed 1-bit Adam

**Key system for OuterLink to learn from:**

- Two-phase approach: warmup with vanilla Adam, then switch to 1-bit compressed communication
- During compression phase: variance is frozen (used as fixed preconditioner), momentum is communicated with error-compensated 1-bit compression
- Results: up to **5x less communication volume**, up to **3.4x faster training** end-to-end
- On Ethernet: up to **6.6x higher throughput** during compression phase
- On InfiniBand: up to **2x higher throughput** during compression phase

#### DeepSpeed 1-bit LAMB

- Extends 1-bit compression to LAMB optimizer (used for large-batch training)
- Up to **4.6x communication volume reduction**
- Up to **2.8x end-to-end speedup** for BERT pre-training
- Compatible with batch sizes 8K-64K

#### DeepSpeed 0/1 Adam (Latest)

- Successor to 1-bit Adam with adaptive variance state freezing
- Up to **26x communication reduction** on BERT-large pre-training
- Allows skipping communication rounds entirely via "1-bit sync"
- More flexible than 1-bit Adam
- Uses NCCL backend (PyTorch distributed)

### 3. Low-Rank Compression

#### PowerSGD

**Core idea:** Approximate the gradient matrix with a low-rank factorization using power iteration.

- Gradient G (m x n) approximated as P (m x r) * Q^T (r x n) where r << min(m,n)
- Communication: send P and Q instead of G
- Compression ratio: mn / (r(m+n)) -- can be very high for large layers

**Performance:**
- Up to **47x fewer bits-per-coordinate** than FP16
- Used by DALL-E training (OpenAI)
- PyTorch has built-in `PowerSGD` communication hook for DDP
- Most effective at lower bandwidths (<8.2 Gbps); above that, uncompressed sync SGD is faster
- Best for large, fully-connected layers; less effective for small or convolutional layers

**Limitations:**
- Requires matrix-shaped gradients (not applicable to biases, norms)
- Power iteration adds compute overhead
- Rank selection is a hyperparameter

### 4. Error Feedback (Error Compensation)

**Not a compression technique itself, but essential for all lossy methods.**

- After compressing gradient, compute the error (difference between original and compressed)
- Add this error to the next iteration's gradient before compression
- Guarantees that no gradient information is permanently lost
- Converts lossy compression into "delayed" lossless compression
- Critical for convergence: without error feedback, 1-bit quantization diverges; with it, convergence matches FP32

### 5. Delta Encoding

**Core idea:** For iterative workloads, send only what changed between iterations.

- XOR-delta: XOR current gradient buffer with previous, compress the delta
- Semantic delta: compute mathematical difference, apply sparsification/quantization to the delta
- Works well when consecutive gradients are similar (which they often are in later training stages)
- Requires maintaining state (previous gradient) on both sender and receiver
- Can be combined with any of the above techniques

## Hybrid Approaches (State of the Art, 2025)

The field is converging on hybrid methods:

| Approach | Techniques Combined | Compression | Notes |
|---|---|---|---|
| **SQ-DeAR** (Euro-Par 2025) | Sparsification + Quantization | Very high | Overlaps communication with forward/backward pass |
| **L-GreCo** (MLSys 2024) | Layerwise-adaptive + PowerSGD | Up to 33% better than kmeans | Per-layer compression selection |
| **ASTC** | Adaptive Sparse Ternary | High | Layer-aware compression ratios |
| **COCCL** (2025) | Quantization + NCCL integration | 3x collective speedup | Built on NCCL 2.21.5 |
| **ghZCCL** (ICS 2025) | Homomorphic compression + collectives | 8.55x vs Cray MPI | Compress-in-place for AllReduce |

### Key Trend: Layerwise Adaptation

Modern approaches don't apply one compression strategy to all layers. Instead:
- Large FC layers: PowerSGD (low-rank) or high sparsification
- Conv layers: moderate sparsification
- Small layers (biases, norms): no compression (overhead > savings)
- Embedding layers: Top-K sparsification (naturally sparse gradients)

## Relevance to OuterLink

### What OuterLink Should Implement

OuterLink operates at the **transport layer**, not the framework layer. It intercepts CUDA calls, not PyTorch/TensorFlow operations. This means:

1. **OuterLink does not know it's compressing gradients.** It sees `cudaMemcpy` of a buffer. It doesn't know the buffer contains gradients.
2. **Semantic compression (Top-K, PowerSGD) must be implemented at the framework level** or via an OuterLink plugin that understands the data semantics.
3. **General-purpose compression (LZ4, nvCOMP) works on any data** and is always applicable.

### Strategy: Two-Tier Compression

**Tier 1 (Transport Layer -- OuterLink Core):**
- General-purpose compression (nvCOMP LZ4, Cascaded) on all VRAM transfers
- Adaptive: skip compression for incompressible data
- No semantic knowledge required
- Handles all CUDA applications

**Tier 2 (Application-Aware -- OuterLink Plugin/NCCL Backend):**
- Gradient-specific compression for ML workloads
- Requires integration with NCCL backend (R20) or framework hooks
- Top-K sparsification, quantization, PowerSGD
- Error feedback state management
- Only active when OuterLink detects collective communication patterns

### NCCL Integration (R20 Cross-Reference)

R20 (NCCL Backend) is where gradient compression becomes powerful. When OuterLink provides its own NCCL backend:

- AllReduce can be implemented with compressed communication
- COCCL and ghZCCL demonstrate this is feasible (3-8.5x speedups)
- Error feedback state can be maintained per-collective
- Layerwise compression can be configured per-operation

This is the **highest-impact feature** for ML workloads and depends on R20 being implemented first.

## Comparison: What Achievable Compression Looks Like

| Method | Compression Ratio | Accuracy Impact | Compute Overhead | Implementation Complexity |
|---|---|---|---|---|
| LZ4 (general) | 2-3x | None (lossless) | Low | Low |
| nvCOMP Cascaded | 20-40x (structured) | None (lossless) | Low (GPU) | Medium |
| Top-K (1%) | 100x | None (with error feedback) | Medium (sorting) | Medium |
| 1-bit quantization | 32x | None (with error feedback) | Low | Medium |
| PowerSGD (rank 4) | 10-50x | Minimal | Medium | High |
| 0/1 Adam | 26x | None | Low | High (optimizer change) |
| Hybrid (sparse + quant) | 100-600x | None (with error feedback) | Medium | High |

## Related Documents

- [R14-01: CPU Compression Algorithms](./01-cpu-compression-algorithms.md) -- General-purpose CPU compression
- [R14-02: GPU-Native Compression (nvCOMP)](./02-gpu-native-compression-nvcomp.md) -- GPU compression engine
- [R20: NCCL Backend](../../R20-nccl-backend/) -- Where gradient compression integrates
- [R14 Pre-Plan](../preplan.md) -- Scope and implementation decisions

## Open Questions

1. **Can OuterLink detect gradient buffers at the transport layer?** Heuristics: buffer size matches layer dimensions, accessed in AllReduce patterns, FP32/FP16/BF16 data types. This would enable Tier 1 to apply gradient-aware heuristics without full semantic knowledge.
2. **Error feedback state management across nodes:** Who owns the error accumulator? In OuterLink's architecture, the server manages GPU state, so error accumulators would live server-side. But they need to be synchronized with the client's view.
3. **Framework-agnostic gradient compression:** Is it possible to implement Top-K or quantization without framework cooperation? Potentially yes, if OuterLink maintains shadow buffers for error feedback.
4. **Convergence validation:** How do we test that compression doesn't degrade ML training quality? Need a benchmark suite of training jobs with known convergence curves.
5. **Interaction with mixed precision:** Modern training uses FP16/BF16 for gradients. Quantizing FP16 to 1-bit is different from quantizing FP32 to 1-bit. Need to handle all precision formats.
6. **Decompression cost at receiver:** For collective operations (AllReduce), every receiver must decompress before accumulating. Does decompression + accumulation cost more than receiving uncompressed + accumulating?
