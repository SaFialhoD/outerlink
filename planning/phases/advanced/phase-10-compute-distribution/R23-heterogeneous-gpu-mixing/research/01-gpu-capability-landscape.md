# GPU Hardware Capability Landscape

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Map the full spectrum of NVIDIA GPU hardware diversity relevant to OuterLink's heterogeneous pool design.

## 1. CUDA Compute Capabilities

Compute Capability (CC) is a major.minor version number that determines what CUDA features a GPU supports and specifies hardware parameters (thread limits, shared memory sizes, instruction throughputs). OuterLink must query CC at runtime to determine kernel compatibility and optimal scheduling.

### Compute Capability by Architecture

| CC | Architecture | Era | Key Consumer GPUs |
|----|-------------|-----|-------------------|
| 3.5 | Kepler | 2013 | GTX 780, GTX Titan |
| 5.0 | Maxwell (1st) | 2014 | GTX 750 Ti |
| 5.2 | Maxwell (2nd) | 2015 | GTX 980, Titan X |
| 6.1 | Pascal | 2016 | GTX 1060/1070/1080, Titan Xp |
| 7.0 | Volta | 2017 | Titan V (rare consumer) |
| 7.5 | Turing | 2018 | RTX 2060/2070/2080, Titan RTX |
| 8.6 | Ampere | 2020 | RTX 3060/3070/3080/3090 |
| 8.9 | Ada Lovelace | 2022 | RTX 4060/4070/4080/4090 |
| 10.0 | Blackwell | 2025 | RTX 5070/5080/5090 |

**Note:** CC 9.0 (Hopper) and 12.0 (Blackwell Ultra) are datacenter-only. OuterLink targets GeForce cards, so the relevant range is **5.2 through 10.0**, with the practical sweet spot being **7.5+ (Turing and later)**.

### Key Feature Introductions by CC

| CC | Feature | Impact on OuterLink |
|----|---------|-------------------|
| 3.5 | Dynamic parallelism, unified memory (basic), Hyper-Q | Minimum for modern CUDA workloads |
| 5.0 | Warp shuffle functions | Required by many kernels |
| 6.1 | FP16 compute, compute preemption | Enables mixed-precision, preemptible kernels |
| 7.0 | Tensor Cores (1st gen), independent thread scheduling | Mixed-precision matrix ops, new scheduling model |
| 7.5 | Tensor Cores (2nd gen) with INT8/INT4 | Quantized inference support |
| 8.0 | TF32/BF16 tensor ops, async copy, L2 persistence, MIG | Modern training/inference features |
| 8.6 | Same as 8.0 minus MIG, 128 CUDA cores/SM | Consumer Ampere (RTX 30 series) |
| 8.9 | Tensor Cores (4th gen) with FP8, thread block clusters (partial) | Quantized LLM inference |
| 9.0 | Thread block clusters, TMA, distributed shared memory | Datacenter only (Hopper) |
| 10.0 | Tensor Cores (5th gen) with FP4/FP6, NVLink 5.0 | Next-gen consumer (Blackwell) |

### Hard Requirements vs Nice-to-Have for OuterLink

**Hard Requirements (minimum CC for participation in pool):**
- CC 5.2+ (Maxwell 2nd gen): warp shuffle, reasonable performance baseline
- Recommended minimum: CC 7.5+ (Turing) for Tensor Core support

**Nice-to-Have (enhance scheduling but not required):**
- FP8 support (CC 8.9+): enables quantized inference on that GPU
- Async copy (CC 8.0+): better overlapping of transfers and compute
- Thread block clusters (CC 9.0+): datacenter-only, not relevant for GeForce

**OuterLink design decision:** The minimum supported CC should be **7.5 (Turing)** for the initial release. This covers RTX 2000/3000/4000/5000 series, which represents the overwhelming majority of GPUs users would actually contribute to a pool. Supporting older Maxwell/Pascal cards adds complexity for negligible benefit.

## 2. GeForce Specifications Comparison

### RTX 30 Series (Ampere, CC 8.6)

| Spec | RTX 3060 | RTX 3070 | RTX 3080 | RTX 3090 |
|------|----------|----------|----------|----------|
| GPU Die | GA106 | GA104 | GA102 | GA102 |
| SMs | 28 | 46 | 68 | 82 |
| CUDA Cores | 3,584 | 5,888 | 8,704 | 10,496 |
| Tensor Cores | 112 (3rd gen) | 184 (3rd gen) | 272 (3rd gen) | 328 (3rd gen) |
| VRAM | 12 GB GDDR6 | 8 GB GDDR6 | 10 GB GDDR6X | 24 GB GDDR6X |
| Bus Width | 192-bit | 256-bit | 320-bit | 384-bit |
| Bandwidth | 360 GB/s | 448 GB/s | 760 GB/s | 936 GB/s |
| Boost Clock | 1.78 GHz | 1.73 GHz | 1.71 GHz | 1.70 GHz |
| TDP | 170W | 220W | 320W | 350W |
| PCIe | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |

### RTX 40 Series (Ada Lovelace, CC 8.9)

| Spec | RTX 4060 | RTX 4070 | RTX 4080 | RTX 4090 |
|------|----------|----------|----------|----------|
| GPU Die | AD107 | AD104 | AD103 | AD102 |
| SMs | 24 | 46 | 76 | 128 |
| CUDA Cores | 3,072 | 5,888 | 9,728 | 16,384 |
| Tensor Cores | 96 (4th gen) | 184 (4th gen) | 304 (4th gen) | 512 (4th gen) |
| VRAM | 8 GB GDDR6 | 12 GB GDDR6X | 16 GB GDDR6X | 24 GB GDDR6X |
| Bus Width | 128-bit | 192-bit | 256-bit | 384-bit |
| Bandwidth | 272 GB/s | 504 GB/s | 717 GB/s | 1,008 GB/s |
| Boost Clock | 2.46 GHz | 2.48 GHz | 2.51 GHz | 2.52 GHz |
| TDP | 115W | 200W | 320W | 450W |
| PCIe | Gen 4 x8 | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |

### RTX 50 Series (Blackwell, CC 10.0)

| Spec | RTX 5070 | RTX 5080 | RTX 5090 |
|------|----------|----------|----------|
| GPU Die | GB205 | GB203 | GB202 |
| SMs | ~48 | ~84 | 170 |
| CUDA Cores | ~6,144 | ~10,752 | 21,760 |
| Tensor Cores | ~192 (5th gen) | ~336 (5th gen) | 680 (5th gen) |
| VRAM | 12 GB GDDR7 | 16 GB GDDR7 | 32 GB GDDR7 |
| Bus Width | 192-bit | 256-bit | 512-bit |
| Bandwidth | ~672 GB/s | ~960 GB/s | 1,792 GB/s |
| Boost Clock | ~2.51 GHz | ~2.62 GHz | 2.41 GHz |
| TDP | 250W | 360W | 575W |
| PCIe | Gen 5 x16 | Gen 5 x16 | Gen 5 x16 |

### RTX 20 Series (Turing, CC 7.5) — Still Relevant for Pools

| Spec | RTX 2060 | RTX 2070 | RTX 2080 | RTX 2080 Ti |
|------|----------|----------|----------|-------------|
| SMs | 30 | 36 | 46 | 68 |
| CUDA Cores | 1,920 | 2,304 | 2,944 | 4,352 |
| Tensor Cores | 240 (2nd gen) | 288 (2nd gen) | 368 (2nd gen) | 544 (2nd gen) |
| VRAM | 6 GB GDDR6 | 8 GB GDDR6 | 8 GB GDDR6 | 11 GB GDDR6 |
| Bandwidth | 336 GB/s | 448 GB/s | 448 GB/s | 616 GB/s |
| PCIe | Gen 3 x16 | Gen 3 x16 | Gen 3 x16 | Gen 3 x16 |

## 3. PCIe Generation Differences

| PCIe Gen | Per-Lane BW | x16 BW | x8 BW | GPUs Using |
|----------|------------|--------|-------|------------|
| 3.0 | ~1 GB/s | ~16 GB/s | ~8 GB/s | RTX 20xx |
| 4.0 | ~2 GB/s | ~32 GB/s | ~16 GB/s | RTX 30xx, 40xx |
| 5.0 | ~4 GB/s | ~64 GB/s | ~32 GB/s | RTX 50xx |

**Impact on OuterLink:** PCIe generation affects host-staged transfer throughput (Phase 1 transport). With OpenDMA (Phase 5), the NIC's PCIe connection matters more than the GPU's. However, for cudaMemcpy-based staging, the GPU's PCIe link is the bottleneck for CPU-GPU transfers.

**Note:** RTX 4060 uses PCIe 4.0 x8 (not x16), giving it effectively the same bandwidth as a PCIe 3.0 x16 card. OuterLink must detect actual PCIe link width and speed, not just the generation.

## 4. BAR1 Size Variations

BAR1 (Base Address Register 1) is the PCIe memory window through which the CPU (and potentially NICs for OpenDMA) can access GPU VRAM.

| Configuration | BAR1 Size | Notes |
|--------------|-----------|-------|
| Default (all GeForce) | 256 MB | Without Resizable BAR enabled |
| ReBAR enabled | = VRAM size | Full VRAM accessible via BAR1 |

**Resizable BAR requirements:** Compatible CPU + motherboard + BIOS (Above 4G Decoding, ReBAR enabled, CSM disabled) + driver R465+ + GPU VBIOS support.

- **RTX 30 series:** Required a VBIOS firmware update to enable ReBAR (except RTX 3060 which shipped with support)
- **RTX 40 series:** ReBAR supported out of the box
- **RTX 50 series:** ReBAR supported out of the box

**Critical for OpenDMA:** Without ReBAR, only 256 MB of VRAM is CPU/NIC-accessible via BAR1, severely limiting direct DMA transfers. OpenDMA effectively requires ReBAR to be enabled on all pool GPUs. OuterLink should detect BAR1 size at registration and warn if it is only 256 MB.

**Multi-GPU BAR1 caveat:** In systems with multiple GPUs, PCIe address space may be insufficient to map full BAR1 for all GPUs simultaneously. A system with 8x RTX 4090 showed 5 GPUs with 32 GB BAR1 but 1 GPU limited to 8 GB due to PCIe topology constraints.

## 5. Tensor Core Generations

| Generation | Architecture | Precision Support | Key Feature |
|-----------|-------------|-------------------|-------------|
| 1st | Volta (CC 7.0) | FP16 mixed | Basic matrix multiply-accumulate |
| 2nd | Turing (CC 7.5) | FP16 + INT8/INT4 | Quantized inference |
| 3rd | Ampere (CC 8.6) | TF32 + BF16 + FP16 | TF32 for easy mixed-precision |
| 4th | Ada (CC 8.9) | FP8 + all above | Efficient transformer inference |
| 5th | Blackwell (CC 10.0) | FP4/FP6 + all above | Ultra-low precision, 2nd gen Transformer Engine |

**OuterLink scheduling implication:** When scheduling a tensor-heavy kernel (e.g., matrix multiply in an LLM), the scheduler should prefer GPUs with later-gen Tensor Cores. A kernel using FP8 must be placed on CC 8.9+ GPUs. A kernel using only FP16 can run on any CC 7.0+ GPU but will be faster on newer Tensor Core generations.

## 6. Runtime Capability Discovery

OuterLink servers must probe each local GPU at startup using the CUDA Driver API:

### Essential Attributes to Query

```
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev)
cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev)
cuDeviceTotalMem(&bytes, dev)  // Total VRAM
cuDeviceGetName(name, len, dev) // GPU model name
```

### Additional Attributes for Scheduling Decisions

```
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE         // Memory clock for bandwidth estimation
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH   // Bus width for bandwidth estimation
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE             // L2 cache size affects memory-bound workloads
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK   // Register pressure limits
CU_DEVICE_ATTRIBUTE_WARP_SIZE                 // Should always be 32 but verify
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS        // Can run multiple kernels simultaneously
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT        // Number of async copy engines
```

### Derived Metrics for OuterLink GPU Profile

From these raw attributes, OuterLink should compute and store:

| Metric | Formula | Purpose |
|--------|---------|---------|
| FP32 TFLOPS | SMs x cores_per_SM x 2 x boost_clock / 1e12 | Raw compute rating |
| Memory BW (GB/s) | mem_clock x bus_width x 2 / 8 / 1e9 | Memory-bound scheduling |
| VRAM (GB) | cuDeviceTotalMem / 1e9 | Capacity planning |
| Compute/BW ratio | TFLOPS / BW | Compute-bound vs memory-bound classification |
| BAR1 size | nvidia-smi or sysfs query | OpenDMA viability |
| PCIe BW | lspci link speed x width | Host-staged transfer rate |

### GPU Profile Structure (Proposed)

```rust
struct GpuProfile {
    device_id: u32,
    name: String,                    // e.g., "NVIDIA GeForce RTX 3090"
    compute_capability: (u32, u32),  // e.g., (8, 6)
    sm_count: u32,
    cuda_cores: u32,                 // sm_count * cores_per_sm (arch-dependent)
    vram_bytes: u64,
    vram_free_bytes: u64,
    memory_bandwidth_gbps: f64,
    fp32_tflops: f64,
    tensor_core_gen: Option<u32>,    // None for pre-Volta
    bar1_size_bytes: u64,
    pcie_gen: u32,
    pcie_width: u32,
    pcie_bandwidth_gbps: f64,
    driver_version: String,
    supports_fp16: bool,
    supports_bf16: bool,
    supports_fp8: bool,
    supports_int8: bool,
    async_engine_count: u32,
    l2_cache_bytes: u32,
}
```

## 7. Cross-Generation Performance Ratios

Understanding relative performance helps the scheduler normalize workloads:

### Approximate FP32 TFLOPS (Theoretical Peak)

| GPU | FP32 TFLOPS | Relative to RTX 3060 |
|-----|------------|---------------------|
| RTX 2060 | 6.5 | 0.51x |
| RTX 2080 Ti | 13.4 | 1.05x |
| RTX 3060 | 12.7 | 1.00x (baseline) |
| RTX 3070 | 20.3 | 1.60x |
| RTX 3080 | 29.8 | 2.35x |
| RTX 3090 | 35.6 | 2.80x |
| RTX 4060 | 15.1 | 1.19x |
| RTX 4070 | 29.1 | 2.29x |
| RTX 4080 | 48.7 | 3.83x |
| RTX 4090 | 82.6 | 6.50x |
| RTX 5090 | ~104.8 | ~8.25x |

**Key observation:** Performance varies by over 16x across the range from RTX 2060 to RTX 5090. A naive round-robin scheduler would be catastrophically inefficient. The scheduler must account for these asymmetries.

### Memory Bandwidth Ratios

| GPU | BW (GB/s) | Relative to RTX 3060 |
|-----|----------|---------------------|
| RTX 2060 | 336 | 0.93x |
| RTX 3060 | 360 | 1.00x |
| RTX 3090 | 936 | 2.60x |
| RTX 4070 | 504 | 1.40x |
| RTX 4090 | 1,008 | 2.80x |
| RTX 5090 | 1,792 | 4.98x |

Memory bandwidth varies by ~5.3x across the range. For memory-bound workloads (most LLM inference), bandwidth matters more than TFLOPS.

## 8. Implications for OuterLink Design

### GPU Registration
Each OuterLink server node must build a `GpuProfile` for every local GPU at startup, transmit it to the cluster coordinator, and update it periodically (VRAM free bytes changes as allocations come and go).

### Kernel Placement Constraints
The scheduler must check CC compatibility before placing a kernel. A kernel compiled for CC 8.9 (using FP8 Tensor Cores) cannot run on a CC 8.6 GPU. PTX JIT can bridge some gaps but cannot provide features the hardware lacks.

### Asymmetric Scheduling
Equal work distribution across different GPUs is wrong by default. The scheduler must distribute work proportional to each GPU's relevant capability (TFLOPS for compute-bound, bandwidth for memory-bound).

### BAR1 Awareness
GPUs without ReBAR enabled are second-class citizens for OpenDMA. They can still participate via host-staged transfers, but direct NIC-to-GPU DMA is limited to the 256 MB BAR1 window, requiring windowed access patterns.

## Related Documents

- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — Memory hierarchy interacts with GPU memory bandwidth
- [R17 Topology-Aware Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/README.md) — GPU capabilities feed into topology map
- [02-heterogeneous-scheduling.md](./02-heterogeneous-scheduling.md) — How to schedule across these different GPUs
- [03-practical-mixing-scenarios.md](./03-practical-mixing-scenarios.md) — Real-world use cases

## Open Questions

- [ ] What is the exact BAR1 behavior on RTX 5090 with ReBAR? Confirm it maps full 32 GB.
- [ ] How does PCIe 5.0 (RTX 50 series) interact with existing ConnectX-5 NICs (PCIe 3.0/4.0)?
- [ ] Should OuterLink support Turing (CC 7.5) at launch, or start at Ampere (CC 8.6) minimum?
- [ ] Tensor Core generation detection at runtime — is there a direct attribute or must we infer from CC?
- [ ] L2 cache size differences (3 MB on RTX 3060 vs 24 MB on RTX 4060) — significant enough to affect scheduling?
