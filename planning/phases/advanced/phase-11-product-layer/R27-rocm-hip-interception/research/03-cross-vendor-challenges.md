# R27 Research: Cross-Vendor GPU Challenges (AMD + NVIDIA)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Analyze the challenges of mixing AMD and NVIDIA GPUs in the same OuterLink pool. Determine whether cross-vendor GPU pooling is realistic, and if so, under what constraints.

---

## TL;DR

Cross-vendor GPU pooling is **realistic but with significant constraints**. The fundamental challenge is ISA incompatibility: CUDA kernels cannot run on AMD GPUs and vice versa. This means a single application cannot transparently use both vendors. However, OuterLink CAN pool both vendors for DIFFERENT applications: CUDA apps use NVIDIA GPUs, HIP apps use AMD GPUs, and the scheduler manages both pools. Same-application cross-vendor execution would require kernel recompilation, which is feasible for HIP apps (HIP compiles for both) but not for pure CUDA apps. The recommended architecture is **per-vendor GPU pools with a unified scheduler**.

---

## 1. ISA Incompatibility

### The Core Problem

| Vendor | Intermediate IR | Native ISA | Wavefront/Warp Size |
|---|---|---|---|
| NVIDIA | PTX | SASS (per-GPU arch: sm_86, sm_89, sm_90) | 32 threads |
| AMD | AMDGCN IL | AMDGCN ISA (per-GPU arch: gfx908, gfx90a, gfx1100) | 64 threads (GCN/CDNA) or 32/64 (RDNA) |

A kernel compiled as PTX/SASS CANNOT execute on AMD hardware. A kernel compiled as AMDGCN CANNOT execute on NVIDIA hardware. There is no runtime translation layer.

### AMD ISA Families

| Family | Architectures | Target Use | Wavefront |
|---|---|---|---|
| GCN (legacy) | GFX7, GFX8 | Older GPUs (R9, RX 400/500) | 64 |
| CDNA | GFX908 (MI100), GFX90A (MI200), GFX940-942 (MI300) | Data center compute | 64 |
| RDNA | GFX1010-1036 (RDNA1/2), GFX1100-1103 (RDNA3), GFX1200 (RDNA4) | Consumer/gaming + compute | 32 or 64 |

Even within AMD, kernels must be compiled for the correct GFX target. An MI300X (gfx942) binary won't run on an RX 7900 XTX (gfx1100).

### NVIDIA ISA Families

| Architecture | Compute Capability | Examples |
|---|---|---|
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |
| Blackwell | sm_100 | B200, RTX 5090 |

### Implication for OuterLink

OuterLink forwards compiled kernel binaries to GPU nodes for execution. It NEVER translates between ISAs. An application must have kernels compiled for the target GPU architecture. OuterLink's role is routing: send the right kernel to the right GPU.

---

## 2. Different Memory Models and Synchronization

### Memory Model Differences

| Aspect | NVIDIA CUDA | AMD HIP/ROCm |
|---|---|---|
| Memory ordering | Weakly ordered, explicit `__threadfence()` | Similar weak ordering, `__threadfence()` equivalent |
| Atomic operations | `atomicAdd`, `atomicCAS`, etc. | `atomicAdd`, `atomicCAS`, etc. (same semantics) |
| Shared memory bank size | 32-bit (configurable to 64-bit) | 32-bit |
| L1/L2 cache | Per-SM L1, shared L2 | Per-CU L1, shared L2 (+ Infinity Cache on RDNA) |
| Unified memory | `cudaMallocManaged`, page migration | `hipMallocManaged`, page migration |
| Pinned memory | `cudaMallocHost` / `cuMemAllocHost` | `hipHostMalloc` |

**Key finding:** Memory semantics are very similar between CUDA and HIP. This is by design since HIP was modeled after CUDA. The differences are in implementation details (cache hierarchy, wavefront size), not in the programming model.

### Synchronization Primitives

| Primitive | CUDA | HIP | Cross-Vendor Issue |
|---|---|---|---|
| Device sync | `cudaDeviceSynchronize` | `hipDeviceSynchronize` | No cross-vendor sync possible |
| Stream sync | `cudaStreamSynchronize` | `hipStreamSynchronize` | Streams are per-vendor |
| Events | `cudaEventRecord/Wait` | `hipEventRecord/Wait` | Events are per-vendor |
| Inter-device sync | CUDA IPC, NVLink | HSA signals, Infinity Fabric | No cross-vendor mechanism |

**Cross-vendor synchronization must go through OuterLink's transport layer.** GPU-to-GPU sync across vendors requires: GPU A completes work -> signals host -> OuterLink transport -> host on GPU B's node -> signals GPU B. This adds latency compared to same-vendor P2P sync.

---

## 3. Unified Memory Pool with Mixed GPU Types

### Architecture: Separate Pools, Unified Scheduler

```
┌──────────────────────────────────────────┐
│         OuterLink Unified Scheduler       │
│  (knows about ALL GPUs, both vendors)     │
├──────────────────┬───────────────────────┤
│  NVIDIA GPU Pool  │    AMD GPU Pool       │
│  (CUDA workloads) │    (HIP workloads)    │
├──────────────────┼───────────────────────┤
│  Node A: 2x 3090 │  Node C: 2x RX 7900  │
│  Node B: 1x 4090 │  Node D: 1x MI100     │
└──────────────────┴───────────────────────┘
```

### Memory Pool Strategies

**Strategy 1: Vendor-Isolated Pools (Recommended for Phase 1)**

Each vendor's GPUs form an independent memory pool. No cross-vendor memory sharing.

| Pros | Cons |
|---|---|
| Simplest implementation | Cannot share data between vendor pools without host bounce |
| No ISA translation needed | Underutilization if one vendor pool is idle |
| Independent fault domains | |

**Strategy 2: Host-Bridged Pools**

Data can be shared between vendor pools by staging through host memory.

```
AMD GPU VRAM -> hipMemcpy -> Host RAM -> cudaMemcpy -> NVIDIA GPU VRAM
```

| Pros | Cons |
|---|---|
| Enables data sharing across vendors | Double copy through host (latency + bandwidth) |
| Useful for pipeline workloads | Host memory becomes bottleneck |
| | Complex to manage |

**Strategy 3: RDMA-Bridged Pools (Future)**

If both vendor pools support RDMA, data can flow: AMD GPU -> RDMA -> Network -> RDMA -> NVIDIA GPU.

```
AMD GPU VRAM --(ROCnRDMA)--> ConnectX-5 --[network]--> ConnectX-5 --(GPUDirect/OpenDMA)--> NVIDIA GPU VRAM
```

| Pros | Cons |
|---|---|
| Zero-host-copy cross-vendor transfer | Complex setup (both vendor RDMA stacks) |
| Maximum bandwidth | Untested combination |
| | Different RDMA mechanisms per vendor |

---

## 4. Scheduling Across Vendors

### Which Workload Goes Where?

| Criterion | Decision Logic |
|---|---|
| Application type | CUDA app -> NVIDIA pool. HIP app -> AMD pool. |
| HIP app (portable) | Can go to either pool. Prefer AMD (native) unless NVIDIA has better perf/availability. |
| Performance requirements | Route to GPU with best compute capability for the workload |
| VRAM requirements | Route to GPU with sufficient free VRAM |
| Availability | If preferred vendor pool is full, queue or reject (don't silently send to wrong vendor) |

### HIP's Cross-Vendor Advantage

HIP applications can compile for BOTH AMD and NVIDIA:
- `hipcc --amdgpu-target=gfx1100` produces AMD binary
- `hipcc --cuda-gpu-arch=sm_86` produces NVIDIA binary (via CUDA backend)

If an application ships both binaries, OuterLink can route it to whichever vendor has available capacity. This is the strongest argument for the HIP ecosystem.

### Scheduling Metadata

Each job submitted to OuterLink must declare:
```
{
  "api": "cuda" | "hip" | "hip-portable",
  "target_architectures": ["sm_86", "gfx1100"],  // which ISAs are available
  "preferred_vendor": "nvidia" | "amd" | "any",
  "vram_required_gb": 20,
  "compute_intensity": "high" | "medium" | "low"
}
```

---

## 5. Driver Coexistence

### Same Node: Both AMD + NVIDIA GPUs

Running both `amdgpu` and `nvidia` kernel drivers on the same node is supported by Linux, but has practical considerations:

| Aspect | Status |
|---|---|
| Kernel drivers | Both `amdgpu` and `nvidia` can load simultaneously |
| Userspace libraries | `libcuda.so` and `libamdhip64.so` coexist without conflict |
| Device nodes | `/dev/nvidia*` (NVIDIA) and `/dev/kfd` + `/dev/dri/*` (AMD) are separate |
| CUDA + HIP apps | Can run simultaneously, each talking to its own driver |
| LD_PRELOAD | Need separate preload libraries per API (`outerlink_cuda.so` vs `outerlink_hip.so`) |
| PCIe | Both GPUs share PCIe bandwidth |
| RDMA | ConnectX-5 can RDMA to both via different peer memory clients |

### Separate Nodes (Recommended)

Each node has GPUs from one vendor only. This is simpler:
- No driver coexistence issues
- No PCIe bandwidth sharing between vendors
- Cleaner RDMA configuration
- Simpler LD_PRELOAD (one library per node)

**Recommendation:** Support both configurations, but RECOMMEND homogeneous nodes. The unified scheduler handles cross-vendor routing across nodes.

---

## 6. Is Cross-Vendor Realistic?

### Realistic Use Cases

| Use Case | Feasibility | Notes |
|---|---|---|
| Separate CUDA and HIP apps sharing a mixed cluster | HIGH | Each app uses its own vendor's GPUs. Scheduler allocates from the right pool. |
| HIP-portable apps using whichever GPU is free | HIGH | HIP compiles for both. OuterLink routes to available GPU. |
| PyTorch with ROCm backend on AMD nodes | HIGH | PyTorch supports ROCm natively. |
| Pure CUDA app running on AMD GPU | IMPOSSIBLE | No CUDA-to-AMDGCN translation exists |
| CUDA app with automatic HIP translation | LOW | HIPIFY can convert source code, but not runtime binaries |
| Unified memory pool across vendors (transparent) | LOW | ISA difference makes transparent mixing impractical |

### Unrealistic Expectations to Avoid

1. **"Drop-in AMD GPU support for CUDA apps"** — Not possible. CUDA apps must be recompiled for HIP/AMD.
2. **"Single GPU that's half NVIDIA, half AMD"** — No. GPUs are discrete units, one vendor each.
3. **"Automatic kernel translation"** — No runtime PTX-to-AMDGCN translator exists or is planned.
4. **"Shared VRAM across vendors"** — Technically possible with host bridging but impractical for most workloads.

### Realistic Value Proposition

**"OuterLink lets you USE all your GPUs, regardless of vendor, for the RIGHT workloads."**

- Have some AMD GPUs from a good deal? Run HIP/ROCm workloads on them.
- Have NVIDIA GPUs for CUDA? Keep using them for CUDA.
- Have HIP-portable code? Let OuterLink put it on whichever GPU is free.
- One scheduler, one pool view, one authentication system, regardless of vendor.

---

## 7. Recommended Architecture

### Phase 1: AMD-Only Pool

First, get HIP interception working for AMD GPUs in an AMD-only cluster. Prove the concept works before mixing vendors.

### Phase 2: Separate Vendor Pools, Unified Scheduler

Add AMD GPU support alongside existing NVIDIA support. Scheduler knows about both pools. Applications declare their API/ISA requirements. No cross-vendor data sharing.

### Phase 3: Cross-Vendor Data Bridging

For pipeline workloads where output from one stage (e.g., on AMD GPU) feeds input to another (on NVIDIA GPU), add host-bridged or RDMA-bridged data transfer between vendor pools.

### Should Nodes Be Homogeneous?

**Recommendation: YES for simplicity, NO as a hard requirement.**

Homogeneous nodes (one vendor per node) simplify:
- Driver configuration
- LD_PRELOAD setup (one library)
- RDMA configuration
- Debugging and troubleshooting

Mixed nodes (both vendors) are possible but add complexity with minimal benefit.

---

## Open Questions

1. **HIP-portable applications:** How many real-world applications actually ship both AMD and NVIDIA binaries? Is this common or theoretical?
2. **Framework support:** PyTorch, TensorFlow, JAX all support ROCm — but how well? Are there performance parity issues?
3. **Cross-vendor benchmark:** What's the actual performance comparison for equivalent workloads on, say, RTX 3090 vs RX 7900 XTX? This affects scheduling decisions.
4. **AMD GPU availability:** Are there enough AMD GPUs in the consumer/prosumer market to make R27 worthwhile? MI100/MI200/MI300 are expensive. Consumer RDNA cards have limited ROCm support.
5. **ROCm on consumer Radeon:** ROCm officially supports only certain Radeon GPUs (RX 7900 series on ROCm 6+). Older cards may not work. This limits the hardware pool.

---

## Related Documents

- [01-rocm-hip-architecture.md](01-rocm-hip-architecture.md)
- [02-interception-feasibility.md](02-interception-feasibility.md)
- [R23: Heterogeneous GPU Mixing](../../phase-10-ecosystem/R23-heterogeneous-gpu-mixing/)
- [R17: Topology-Aware Scheduling](../../phase-09-distributed-os/R17-topology-aware-scheduling/)
