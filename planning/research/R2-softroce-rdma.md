# R2: SoftRoCE / rdma_rxe - Capabilities and Limitations

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** HIGH

## Purpose

Evaluate SoftRoCE (rdma_rxe) as the core transport layer for OutterLink GPU memory sharing.

## TL;DR - VERDICT: NOT SUITABLE AS PRIMARY TRANSPORT

SoftRoCE has critical limitations that disqualify it as our production transport:
1. **Throughput is ~30% of TCP** on the same link (WORSE, not better)
2. **Cannot do GPUDirect RDMA** - no DMA engine, all goes through CPU
3. **Stability issues** - crashes after ~1000 connections
4. **Deprecated in RHEL** - being removed in RHEL 10

However, SoftRoCE is useful for **development/testing** if we later want hardware RDMA compatibility.

---

## What Is SoftRoCE

- Kernel module `rdma_rxe` implementing RoCE v2 protocol entirely in software
- Encapsulates InfiniBand transport inside UDP packets (port 4791)
- In Linux kernel since 4.8 (2016)
- Provides standard InfiniBand verbs API - any IB verbs app works with it
- Works over any Ethernet NIC, any switch, no special config

## Performance Reality Check

### Throughput - WORSE Than TCP

| Setup | Transport | Throughput |
|-------|-----------|-----------|
| Same link | **TCP** | **~300 MB/s (~2.4 Gbps)** |
| Same link | **SoftRoCE** | **~100 MB/s (~0.8 Gbps)** |
| 25GbE link | TCP | 19.2 Gbps |
| 25GbE link | SoftRoCE | Much less than 19.2 Gbps |
| Hardware RoCE (ConnectX) | Hardware RDMA | ~3,323 MB/s |

**Why so slow:** SoftRoCE is capped at 4KiB MTU per packet while TCP uses 64KiB with TSO/GSO. Plus software CRC32 computation on every packet burns CPU.

### Latency

| Transport | Round-trip Latency |
|-----------|-------------------|
| Hardware RDMA | ~1.3 us |
| TCP/IP | ~10-15 us |
| SoftRoCE | **~52 us** |

SoftRoCE latency is **40-50x worse** than hardware RDMA and **3-5x worse** than plain TCP.

### CPU Overhead

All packet processing on host CPU. CRC32 calculation is the main bottleneck. No hardware offload whatsoever. Fundamentally different from hardware RDMA where the NIC DMA engine handles transfers.

## GPU Memory - CRITICAL LIMITATION

**SoftRoCE CANNOT do GPUDirect RDMA.**

- GPUDirect requires a hardware NIC with its own DMA engine
- `nvidia-peermem` kernel module only works with hardware HCAs (ConnectX-3+)
- SoftRoCE has no DMA engine - everything goes through kernel networking stack

**What IS possible:** Register CUDA pinned host memory (`cudaMallocHost`) as RDMA buffers, then `cudaMemcpy` between GPU VRAM and that host buffer. But this adds an extra copy through host memory.

## Stability Issues

- Stops responding / crashes after ~1000 connections
- Double free in `mr->map` (fixed in kernel 6.0.16+)
- NULL pointer dereferences under certain conditions
- Use-after-free race condition (CVE-2025-40061)
- Corrupted retransmissions under stress with many QPs
- NVMe-oF over SoftRoCE: multiple reports of kernel crashes
- Device disappears shortly after creation (multiple Ubuntu reports)

## RXE-to-RXE Optimizations (Break Hardware Interop)

When only talking to other SoftRoCE nodes:
- Disable ICRC: `crc_disable=1` in sysfs
- Fast tasklet mode: `fast_req=1`, `fast_resp=1`, `fast_comp=1`
- With kernel patches: increase MTU above 4KiB
- Loopback with all optimizations: ~35 GB/s (not realistic over network)

## Setup (For Reference)

```bash
modprobe rdma_rxe
rdma link add rxe0 type rxe netdev eth0
rdma link
ibv_devices
```

Works over regular switches, standard IP addressing, no special network config. Does NOT survive reboots.

---

## Alternatives - RANKED BY SUITABILITY

### 1. Well-Tuned TCP + CUDA Pinned Memory (RECOMMENDED START)
- Simpler, faster than SoftRoCE, battle-tested
- Pipeline: `cudaMemcpy GPU->pinned_host` -> TCP send -> TCP recv -> `cudaMemcpy pinned_host->GPU`
- Use `io_uring` for lower syscall overhead
- **Verdict: Best starting point. Simple, fast, reliable.**

### 2. DPDK (User-Space Kernel Bypass)
- Bypasses kernel entirely, processes packets in user-space
- Can match RDMA latency for small messages
- Works with any NIC via VFIO
- Requires dedicated CPU cores (poll mode)
- Must reimplement reliability, congestion control
- **Verdict: Highest performance ceiling without hardware. High development cost.**

### 3. io_uring for Network I/O
- Linux async I/O framework (kernel 5.1+)
- Batched syscall submission/completion
- Zero-copy RX being developed
- Works with standard NICs
- Not RDMA but significantly reduces syscall overhead
- **Verdict: Great complement to TCP approach. Moderate development cost.**

### 4. UCX (Unified Communication X)
- Abstraction layer over multiple transports (TCP, RDMA, shared memory, CUDA)
- Used by MPI implementations and ML frameworks
- Can start with TCP, upgrade to hardware RDMA later transparently
- **Verdict: Good abstraction if we want transport flexibility.**

### 5. Soft-iWARP (siw)
- Similar to SoftRoCE but uses TCP instead of UDP
- Same performance ballpark
- Also deprecated in RHEL
- **Verdict: No advantage over SoftRoCE. Skip.**

### 6. Hardware RoCE (ConnectX) - Future Path
- True GPUDirect RDMA, zero-copy GPU-to-GPU
- 1-2us latency, full line-rate throughput
- Requires $500-2000+ cards
- **Verdict: The "if we get funding" upgrade path. Design for it but don't require it.**

---

## Impact on OutterLink

### What This Changes

1. **SoftRoCE should NOT be our primary transport** - TCP is faster and more stable
2. **Our transport layer should be abstracted** - start with TCP, allow upgrading to RDMA hardware later
3. **GPU memory transfer always goes through host memory** without hardware RDMA - this is the same overhead whether we use SoftRoCE or TCP
4. **The project name "OutterLink" and concept still works** - we're just using a different wire protocol

### Revised Architecture Implication

```
GPU VRAM (remote)
    -> cudaMemcpy to pinned host memory
    -> TCP/io_uring send over network
    -> TCP/io_uring recv on local machine
    -> cudaMemcpy from pinned host memory
    -> GPU VRAM (local) or process use
```

The extra hop through host memory happens regardless of transport choice (without ConnectX + GPUDirect). So TCP vs SoftRoCE doesn't add any extra copy - it just determines network layer performance, where TCP wins.

## Related Documents

- [Project Vision](../../docs/architecture/00-project-vision.md)
- [Pre-Planning Master](../pre-planning/00-master-preplan.md)

## Sources

1. rxe(7) Linux manual page
2. Red Hat RHEL 7: Configuring Soft-RoCE (benchmark numbers)
3. rdma-core Documentation: rxe.md (upstream docs)
4. Linux RDMA mailing list: throughput analysis
5. GPUDirect RDMA Documentation (NVIDIA)
6. soft-roce-proxmox-lxc benchmarks (GitHub)
7. SoftRoCE original whitepaper (RoCE Initiative)
8. CVE-2025-40061 advisory
9. IEEE 2025: RDMA over Soft-RoCE with eBPF Monitoring

## Open Questions

- [x] Is SoftRoCE suitable as primary transport? **NO**
- [ ] Should we use UCX as abstraction layer for transport flexibility?
- [ ] What io_uring features are stable enough for production use?
- [ ] Can DPDK complexity be justified for v1, or is it a v2 optimization?
