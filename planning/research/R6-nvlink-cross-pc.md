# R6: NVLink as Cross-PC Bridge

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** HIGH

## Purpose

Evaluate whether NVLink can be used to connect GPUs across separate physical PCs for 600 GB/s inter-node bandwidth.

---

## Hardware Clarification: RTX 3090 Ti NVLink

Both the RTX 3090 and RTX 3090 Ti support NVLink 3.0. They are the only GeForce 30-series cards with NVLink support. The connector was removed starting with the RTX 40 series.

---

## Can NVLink Bridge Across PCs?

**Short answer: No, not with any existing technology.**

### Physical Constraints

NVLink bridges are ~6-10cm PCB connectors designed for GPUs in adjacent slots on the same motherboard. The signaling specs make extension extremely challenging:

| Parameter | Value |
|-----------|-------|
| Signaling rate | 50 Gbit/s per differential pair |
| Pairs per direction | 8 differential pairs |
| Total wires | 32 (bidirectional) |
| Coupling | DC-coupled, 85-ohm differential |
| Clock | Embedded (no separate clock line) |

At these speeds, signal integrity degrades rapidly with distance:

| Cable Technology | Max Distance | Notes |
|-----------------|-------------|-------|
| Passive copper (DAC) | ~1-2 meters | Too lossy for NVLink 3.0 |
| Active copper (ACC) | ~1.5 meters | Used in GB200 NVL72 enterprise |
| Active enhanced copper (AEC) | ~5-6 meters | Signal reconstruction via retimers |
| Optical (not yet available) | Kilometers | NVIDIA researching for Rubin architecture |

Nobody has built or demonstrated a consumer NVLink cable extender. The enterprise GB200 NVL72 uses ~5,000 active copper cables and NVSwitch ASICs to connect 72 GPUs across racks - this is a $millions system, not something reproducible.

### Connector Limitation (RTX 3090)

Even if the non-Ti 3090 is used:
- **1 NVLink connector per GPU = 1 peer only**
- Cannot daisy-chain: GPU_A -- GPU_B -- GPU_C requires GPU_B to have 2 connectors
- Cannot split or multiplex a single NVLink connection
- No consumer NVLink switch exists

So the topology `PC1:GPU_A <-NVLink-> PC2:GPU_B` would use the ONLY NVLink connector on each GPU. No local NVLink pairs possible simultaneously.

### Software Barriers (Even If Physical Connection Existed)

| Barrier | Detail |
|---------|--------|
| PCIe domain | Driver expects NVLink peers in same PCIe domain. Cross-PC = separate domains. |
| Fabric Manager | Required for NVSwitch systems (DGX H100+). Not available for consumer. |
| IMEX daemon | CUDA 12.4+ inter-node NVLink needs IMEX. Enterprise only. |
| IOMMU | CUDA doesn't support IOMMU-enabled PCIe P2P. Separate machines = separate IOMMU. |
| `cudaDeviceCanAccessPeer()` | Checks PCIe/NVLink topology. Would fail for cross-machine GPUs. |

### Bandwidth Reality: RTX 3090 NVLink vs Datacenter

| GPU | NVLink Version | Links | Aggregate Bandwidth |
|-----|---------------|-------|-----------|
| **RTX 3090 / 3090 Ti** | **NVLink 3.0** | **4 x4 links** | **112.5 GB/s** |
| A100 | NVLink 3.0 | 12 links | ~600 GB/s |
| H100 | NVLink 4.0 | 18 links | ~900 GB/s |

The RTX 3090/3090 Ti NVLink provides 112.5 GB/s aggregate bandwidth between a pair of GPUs (56.25 GB/s bidirectional per direction). This is ~2x what 4x 100GbE bonded can achieve (~50 GB/s).

### ConnectX-5 vs NVLink Comparison

| Connection | Bandwidth | Notes |
|-----------|-----------|-------|
| RTX 3090 NVLink | 112.5 GB/s | Local GPU pair, same machine |
| 4x 100GbE bonded | ~50 GB/s | Cross-PC, already owned |
| 2x 100GbE bonded | ~25 GB/s | Cross-PC, simpler setup |

NVLink is ~2x faster than our best network option. Both are valuable: NVLink for local GPU pairs, ConnectX-5 for cross-PC.

---

## NVSwitch: Not An Option

| Fact | Detail |
|------|--------|
| What it is | Physical ASIC chip, 144 NVLink ports, 14.4 TB/s switching |
| Available separately? | **NO** - only in DGX, HGX, GB200 NVL72 |
| Third-party alternatives? | **None exist** |
| Open-source alternatives? | **None exist** |
| UALink (open standard) | Planned for late 2026+, not available yet |

---

## Alternatives for High-Speed Cross-PC GPU Connection

| Technology | Bandwidth | Availability | Cost |
|-----------|-----------|-------------|------|
| **4x 100GbE bonded (ConnectX-5)** | **~50 GB/s** | **Have it!** | **Already owned** |
| InfiniBand HDR | ~25 GB/s | Used market, $50-150/card | Affordable |
| GigaIO FabreX (PCIe fabric) | ~64 GB/s (PCIe 4.0) | Enterprise only | $$$ |
| InfiniBand NDR | ~50 GB/s | New, expensive | $$$ |
| OCuLink (PCIe over cable) | ~8 GB/s (PCIe 4.0 x4) | Consumer products | $ |

---

## The Strategy: NVLink + ConnectX-5 Together

NVLink and ConnectX-5 are complementary, not competing:

```
PC1: GPU_A <--NVLink 112.5 GB/s--> GPU_B
                                        \
                                    ConnectX-5: 4x 100GbE (~50 GB/s)
                                        /
PC2: GPU_C <--NVLink 112.5 GB/s--> GPU_D
```

- **NVLink:** Fast local GPU pairs within each PC (112.5 GB/s)
- **ConnectX-5:** Cross-PC connection (up to ~50 GB/s bonded)
- **OutterLink software:** Makes the entire pool appear unified

This gives a two-tier bandwidth architecture:
- Local GPU pair operations: 112.5 GB/s (NVLink)
- Cross-PC operations: ~50 GB/s (ConnectX-5)
- Smart scheduling can prefer local GPU pairs for bandwidth-heavy work

---

## Verdict

| Approach | Feasible? | Worth Pursuing? |
|----------|-----------|----------------|
| NVLink bridge across PCs | NO (physical, software barriers) | NO |
| NVSwitch for consumer GPUs | NO (not sold separately) | NO |
| Custom NVLink cable extender | Theoretically possible, practically extreme | NO |
| 4x 100GbE ConnectX-5 bonded | YES (already owned!) | **YES - equivalent bandwidth** |
| GigaIO FabreX | YES but enterprise-priced | MAYBE (future) |

**Recommendation:** Focus on maximizing ConnectX-5 performance. You already have hardware that matches RTX 3090 NVLink bandwidth. The software (OutterLink) is what's missing, and that's what we're building.

---

## Future: UALink and Optical Interconnects

- **UALink** - Open standard alternative to NVLink, expected late 2026+. Backed by AMD, Intel, Google, Microsoft, Meta. Could democratize high-speed GPU interconnects.
- **Optical NVLink** - NVIDIA researching silicon photonics (Ayar Labs) for Rubin architecture (2026-2027). Would enable NVLink over fiber optic at kilometer distances.
- **CXL 3.0** - Memory pooling standard. Not GPU-to-GPU, but could enable shared memory pools that GPUs access.

These are worth monitoring but not actionable today.

## Related Documents

- [R4: ConnectX-5 + Transport](R4-connectx5-transport-stack.md)
- [R5: GPUDirect on GeForce](R5-gpudirect-geforce-restriction.md)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)

## Sources

1. NVIDIA Developer Forums - NVLink port support for RTX 3090 Ti confirmation
2. NVIDIA NVLink Bridges Product Page
3. NVLink Wikipedia / WikiChip specifications
4. GB200 NVL72 Hardware Architecture (SemiAnalysis)
5. NVIDIA Cable Management Guidelines
6. GigaIO FabreX PCIe Memory Fabric documentation
7. CUDA Programming Guide - Multi-GPU Systems
8. NVIDIA Fabric Manager User Guide
9. UALink and CXL interconnect analysis
