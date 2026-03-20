# Hardware Inventory

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Draft - Needs confirmation on some details

## Purpose

Document all available hardware for OutterLink development and testing.

---

## PC1: Minisforum MS-01 Ultra

| Component | Spec | Notes |
|-----------|------|-------|
| **CPU** | TBD | MS-01 Ultra |
| **RAM** | 256 GB | Dual channel (~90 GB/s bandwidth) |
| **PCIe** | PCIe 5.0 | |
| **NICs** | 2x ConnectX-5 100GbE (dual port each) | 4 total 100GbE ports |
| **GPUs** | TBD (from pool below) | Via risers, open air |
| **OS** | TBD | |

### RAM Note
Dual channel at ~90 GB/s still exceeds PCIe 4.0 x16 (32 GB/s). PCIe remains the GPU bottleneck, not RAM.

---

## PC2: Threadripper Workstation

| Component | Spec | Notes |
|-----------|------|-------|
| **CPU** | AMD Threadripper 9960X | 24 cores, Zen 5 |
| **Motherboard** | TRX50 chipset | |
| **RAM** | 256 GB ECC DDR5 4800 | ECC for reliability |
| **PCIe** | PCIe 5.0 | |
| **NICs** | 2x ConnectX-5 100GbE (dual port each) | 4 total 100GbE ports |
| **GPUs** | TBD (from pool below) | Via risers, open air |
| **OS** | TBD | |

---

## GPU Pool

| GPU | Count | VRAM | PCIe Gen | NVLink | Notes |
|-----|-------|------|----------|--------|-------|
| RTX 3090 Ti | 2 | 24 GB GDDR6X each | 4.0 x16 | 3rd gen (bridge needed) | Open air with risers for NVLink fit |
| RTX 5090 | 2 | 32 GB GDDR7 each | 5.0 x16 | TBD | Newest gen |

**Total VRAM:** 2x24 + 2x32 = **112 GB**
**Total system RAM:** 256 + 256 = **512 GB**
**Total combined pool:** 112 GB VRAM + 512 GB system RAM = **624 GB**

### GPU Placement Strategy (TBD)
Option A: One 3090 Ti + one 5090 per PC (mixed)
Option B: Both 3090 Ti in one PC (NVLink pair), both 5090 in other
Option C: Other arrangement

NVLink pairing requires same GPU model. So NVLink pairs would be:
- 3090 Ti <-> 3090 Ti (if in same PC)
- 5090 <-> 5090 (if in same PC, and if 5090 supports NVLink)

---

## Network Infrastructure

| Component | Spec | Count |
|-----------|------|-------|
| ConnectX-5 100GbE cards | Dual port | 4 cards total (2 per PC) |
| Total 100GbE ports | | 8 ports (4 per PC) |

### Network Topology Options

**Option A: Direct cable (simplest)**
```
PC1 [CX5 port] ----DAC/fiber---- [CX5 port] PC2
```
Single 100 Gbps link = ~12.5 GB/s

**Option B: Bonded direct cables**
```
PC1 [CX5 port 1] ----DAC---- [CX5 port 1] PC2
PC1 [CX5 port 2] ----DAC---- [CX5 port 2] PC2
```
Bonded 200 Gbps = ~25 GB/s (with LACP or round-robin bonding)

**Option C: Multi-link (4 cables, all ports)**
```
PC1 [4 ports] ----4x DAC---- [4 ports] PC2
```
Up to 400 Gbps = ~50 GB/s (approaches PCIe 4.0 x16 bandwidth!)

### Bandwidth Comparison with Multi-Link

| Connection | Bandwidth | vs PCIe 4.0 x16 |
|-----------|-----------|-----------------|
| NVLink 3090 Ti | ~600 GB/s | 19x faster |
| PCIe 5.0 x16 | ~64 GB/s | 2x faster |
| PCIe 4.0 x16 (3090 Ti native) | ~32 GB/s | Baseline |
| **4x 100GbE bonded** | **~50 GB/s** | **1.5x faster!** |
| 2x 100GbE bonded | ~25 GB/s | 78% of PCIe 4.0 |
| 1x 100GbE | ~12.5 GB/s | 39% of PCIe 4.0 |
| PCIe 4.0 x4 riser | ~8 GB/s | 25% |
| PCIe 4.0 x1 riser | ~4 GB/s | 12.5% |

**KEY INSIGHT:** With 4x bonded 100GbE links, network bandwidth EXCEEDS a single PCIe 4.0 x16 connection. A remote GPU over 4x100GbE could be faster than a local GPU on a x4 riser. With 2x bonded, it's close to native PCIe 4.0.

---

## ConnectX-5 100GbE Capabilities

| Feature | Supported | Notes |
|---------|-----------|-------|
| RoCE v2 (RDMA over Ethernet) | YES | Hardware RDMA |
| GPUDirect RDMA | YES (hardware) | **BUT blocked by NVIDIA driver for GeForce GPUs** |
| DPDK kernel bypass | YES | Available but probably overkill |
| Bonding/LACP | YES | Multi-link aggregation |
| SR-IOV | YES | Virtual functions for isolation |
| PTP timestamping | YES | Precise timing |

### GPUDirect RDMA Status: RESEARCHING

NVIDIA restricts GPUDirect RDMA to Data Center and Professional GPUs at the driver level. Research in progress on:
- Exact restriction mechanism (driver check? firmware?)
- Whether nvidia-open kernel modules change anything
- Community workarounds
- Legal implications

---

## PCIe Topology Considerations

Both PCs have PCIe 5.0. For optimal performance:
- ConnectX-5 cards should be in PCIe slots that share a root complex with the GPUs
- Run `lspci -tv` and `nvidia-smi topo -m` when machines are set up to verify
- Suboptimal topology (GPU and NIC on different root complexes) can reduce RDMA bandwidth by 10x

---

## Open Questions

- [ ] Which GPUs go in which PC?
- [ ] Do 5090s support NVLink? (Need to verify)
- [ ] What riser type/lane count for open air setup?
- [ ] What OS/distro on each machine?
- [ ] What CUDA version will be installed?
- [ ] Direct cables or through a switch?
- [ ] What SFP modules / DAC cables needed for 100GbE?
- [ ] Can GPUDirect RDMA be enabled on GeForce? (Research in progress)

## Related Documents

- [Project Vision](../../docs/architecture/00-project-vision.md)
- [R4: ConnectX-5 + Transport Stack](../../planning/research/R4-connectx5-transport-stack.md)
