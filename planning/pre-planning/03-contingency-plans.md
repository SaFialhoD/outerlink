# Contingency Plans: Plan B for Every Critical Path

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete

## Purpose

For every critical technical decision, document what we do if Plan A fails. We should never be stuck without a next move.

---

## 1. OpenDMA (Non-Proprietary GPU DMA)

**This is our highest-risk, highest-reward feature.** Multiple fallback paths are essential.

| Priority | Approach | How It Works | Risk Level | What Could Block It |
|----------|----------|-------------|-----------|-------------------|
| **Plan A** | tinygrad BAR1 patches + custom RDMA module | Patch nvidia-open to map all VRAM into BAR1, write kernel module to register BAR1 with RDMA subsystem | MEDIUM | GSP firmware rejects it; tinygrad patches don't work on 3090 Ti specifically |
| **Plan B** | nouveau upstream RDMA patches | Use NVIDIA engineer's patches (RFC Dec 2024, v2 July 2025) for GPU RDMA via nouveau driver | MEDIUM | Patches not merged yet; nouveau has no CUDA compute |
| **Plan C** | Linux P2PDMA framework | Write custom P2PDMA provider module that registers GPU BAR1 as P2P resource | MEDIUM | Needs GPU MMU setup from another source |
| **Plan D** | VFIO + DMA-BUF (Linux 6.19+) | Use VFIO to claim GPU BAR1, export as DMA-BUF, import in RDMA NIC | LOW | Requires exclusive GPU access (no NVIDIA driver simultaneously) |
| **Plan E** | Host-staged forever | cudaMemcpy to pinned host -> RDMA/TCP -> cudaMemcpy back | WORKS TODAY | Not zero-copy, ~12us vs ~2us latency. Still viable. |

### Decision Tree

```
Can we apply tinygrad patches on our 3090 Ti?
├── YES: Does nvidia_p2p_get_pages() work after patching?
│   ├── YES: OpenDMA Plan A WORKS! Use nvidia-peermem as-is.
│   └── NO: Write custom peer memory client for BAR1 → Plan A variant
│       ├── WORKS: Ship it!
│       └── FAILS (GSP firmware blocks): → Plan B (nouveau patches)
│           ├── Patches merged + VRAM export works: Ship it!
│           └── Not merged / no CUDA: → Plan C (P2PDMA)
│               ├── WORKS: Ship it!
│               └── FAILS: → Plan E (host-staged, still ships, just slower)
└── NO (patches don't compile/apply): → Try aikitoria fork → Plan B/C/E
```

### Key Validation Tests (In Order)

1. Apply tinygrad patches, boot system, verify GPU still works with CUDA
2. Check BAR1 size: `nvidia-smi -q | grep BAR1` (should show full VRAM)
3. Try `nvidia-peermem` on GeForce with patched driver
4. If that fails: load our custom module, attempt RDMA write to BAR1 address
5. Verify data integrity: write pattern to GPU via RDMA, read back via cudaMemcpy

---

## 2. CUDA Interception

| Priority | Approach | How It Works | Risk Level | What Could Block It |
|----------|----------|-------------|-----------|-------------------|
| **Plan A** | Driver API + LD_PRELOAD + cuGetProcAddress | Our primary approach (HAMi-core pattern, 222 funcs) | LOW | Extremely unlikely - this is proven tech |
| **Plan B** | Runtime API interception | Intercept libcudart.so instead of libcuda.so | LOW | Misses apps using Driver API directly; static linking bypasses |
| **Plan C** | Fork/adapt gVirtuS frontend | Use gVirtuS's split-driver model, replace backend with our transport | MEDIUM | GPL license contamination (we're Apache 2.0) |
| **Plan D** | Fork/adapt SCUDA's code-gen approach | Auto-generate RPC stubs from CUDA headers | MEDIUM | SCUDA is early stage, may need heavy modification |

### When to Fall Back

- Plan A fails if: cuGetProcAddress behavior changes radically in a future CUDA version (unlikely)
- Plan B is a simplification, not a failure path - useful if we want faster PoC
- Plan C/D are "build on existing code" options if clean-room is too slow

**Realistically: Plan A will work.** This is the most battle-tested part of our design. HAMi-core runs in production with 10,000+ Kubernetes pods.

---

## 3. Transport Layer

| Priority | Approach | Bandwidth (100GbE) | Complexity | What Could Block It |
|----------|----------|-------------------|-----------|-------------------|
| **Plan A** | TCP + io_uring zero-copy | ~10 GB/s | LOW | io_uring ZC needs Linux 6.15+ |
| **Plan B** | Plain TCP (tokio) | ~8 GB/s | VERY LOW | Nothing - works everywhere |
| **Plan C** | UCX (auto RDMA/TCP) | ~12 GB/s (RDMA) | MEDIUM | UCX Rust bindings may be immature |
| **Plan D** | Raw libibverbs (`sideway` crate) | ~12 GB/s | HIGH | Complex API, error-prone |
| **Plan E** | Plain TCP + manual RDMA for bulk | ~10-12 GB/s | MEDIUM | Two code paths to maintain |

### Decision Tree

```
Is Linux kernel >= 6.15?
├── YES: Use TCP + io_uring zero-copy (Plan A)
│   └── Later: Add UCX backend for RDMA (Plan C)
└── NO: Use plain tokio TCP (Plan B)
    └── Later: Upgrade kernel, switch to Plan A
```

### Key Point

Transport is the **lowest risk** area. TCP works everywhere. Everything else is optimization. We should NOT let transport choice block progress.

---

## 4. Rust FFI with CUDA

| Priority | Approach | Pros | Cons |
|----------|----------|------|------|
| **Plan A** | `cudarc` crate | Active development, safe wrappers, community | May not cover all 222 Driver API functions |
| **Plan B** | Raw `libloading` + hand-written FFI | Full control, covers everything | More boilerplate, no safety wrappers |
| **Plan C** | Interception .so in pure C, Rust for server only | Proven approach (HAMi-core is C), simplest FFI | Two languages to maintain |
| **Plan D** | `bindgen` from CUDA headers | Auto-generated, complete | Generated code may need cleanup |

### Recommended Approach

Start with **Plan C** (C interception .so + Rust server). This is the safest because:
- The .so needs to be loaded via LD_PRELOAD - C is the natural fit
- HAMi-core, Cricket, SCUDA are all C - we can directly reference their patterns
- The Rust server does the heavy lifting (transport, scheduling, memory management)
- Clean language boundary: C handles CUDA interception, Rust handles everything else

Fall back to Plan A/B only if we find a way to write the .so directly in Rust with acceptable FFI overhead.

---

## 5. Hardware / Setup Contingencies

| Situation | Plan B |
|-----------|--------|
| PCIe topology: GPU and NIC on different root complexes | Rearrange cards in different slots; worst case: P2P works but slower |
| ReBAR not supported on our motherboard | OpenDMA limited to 256MB BAR1 window (usable but requires windowing logic) |
| NVLink bridge doesn't fit with risers | Skip NVLink, rely entirely on ConnectX-5 network (~50 GB/s bonded still excellent) |
| 3090 Ti actually doesn't have NVLink connector | Buy non-Ti 3090s (cheaper anyway) OR skip NVLink |
| MLNX_OFED conflicts with NVIDIA driver | Use inbox kernel mlx5 driver (less features but works) |
| ConnectX-5 only does 25GbE (wrong SFP) | Buy correct QSFP28 modules; still viable at 25GbE (~3 GB/s) |

---

## 6. Project-Level Contingencies

| Situation | Plan B |
|-----------|--------|
| Rust FFI is too painful for the interception layer | Write interception in C, keep server in Rust |
| Performance is not competitive with gVirtuS/TensorFusion | Focus on OpenDMA as differentiator - no one else has it |
| NVIDIA releases an official remote GPU API | Pivot to wrapper/enhancement of official API + OpenDMA |
| A competing open-source project appears | Differentiate with OpenDMA + Rust + Apache 2.0 |
| Project scope is too large for two people | Release Phase 1-2 as standalone tool, grow community, then tackle Phase 3+ |

---

## Summary: What's Actually Risky vs What's Not

### LOW RISK (Plan A will almost certainly work)
- CUDA interception (proven by HAMi-core, Cricket, SCUDA, gVirtuS)
- Transport layer (TCP works, everything else is optimization)
- Basic GPU remoting concept (proven by 30+ projects)

### MEDIUM RISK (Plan A might need adjustment)
- Rust FFI with CUDA (might need C for .so, Rust for server)
- OpenDMA on 3090 Ti specifically (tinygrad tested on 4090, not 3090 Ti)
- UCX Rust bindings maturity

### HIGH RISK (but with solid fallbacks)
- OpenDMA bypassing GSP firmware (unknown if firmware independently blocks)
- Full CUDA app compatibility (222 functions is a lot, edge cases exist)

### THE ONE TRUE UNKNOWN
**GSP firmware behavior with OpenDMA.** We won't know until we test on real hardware. If the GSP independently blocks BAR1 peer access regardless of driver patches, Plan A-C for OpenDMA all fail. Plan E (host-staged) always works but loses the zero-copy advantage.

This is why we build Phase 1-4 (host-staged) FIRST and OpenDMA (Phase 5) SECOND. The project ships and is useful even without OpenDMA.

## Related Documents

- [FINAL PRE-PLAN](02-FINAL-PREPLAN.md)
- [R7: Non-Proprietary GPU DMA](../research/R7-non-proprietary-gpu-dma.md)
- [R3: CUDA Interception](../research/R3-cuda-interception.md)
- [R4: ConnectX-5 + Transport](../research/R4-connectx5-transport-stack.md)
