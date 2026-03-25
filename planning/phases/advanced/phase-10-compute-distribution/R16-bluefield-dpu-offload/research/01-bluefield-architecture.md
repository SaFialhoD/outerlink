# R16 Research: BlueField DPU Architecture

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** APPROVED
**Purpose:** Document BlueField hardware capabilities, architecture, and suitability for OuterLink transport offload.

---

## 1. What Is a DPU?

A Data Processing Unit (DPU) is a programmable network adapter that combines:
- **ARM SoC** (general-purpose CPU cores + DRAM)
- **ConnectX NIC** (RDMA-capable network adapter)
- **PCIe switch** (sits between host PCIe bus and network)
- **Hardware accelerators** (compression, crypto, regex, packet processing)

The DPU runs its own Linux OS independently from the host. It intercepts all network traffic between the host and the wire, processing or forwarding it via its ARM cores and hardware engines. The host sees the DPU as a regular NIC, but the DPU can inspect, modify, route, compress, or encrypt every packet before it reaches the host — or without the host ever seeing it at all.

Originally developed by Mellanox Technologies (acquired by NVIDIA in 2019 for $6.9B).

---

## 2. BlueField-2 vs BlueField-3 Specifications

| Feature | BlueField-2 | BlueField-3 |
|---|---|---|
| **ARM Cores** | 8x Cortex-A72 @ 2.0-2.5 GHz | 16x Cortex-A78 @ 3.0 GHz |
| **Core Microarchitecture** | ARMv8, in-order (A72) | ARMv8.2, out-of-order (A78) |
| **On-board DRAM** | 16 GB or 32 GB DDR4 @ 3200 MT/s | 32 GB DDR5 ECC |
| **Embedded NIC** | ConnectX-6 Dx | ConnectX-7 |
| **Network Speed** | Up to 2x100 GbE or 1x200 GbE | Up to 2x200 GbE or 1x400 GbE |
| **InfiniBand** | HDR (200 Gb/s) | NDR (400 Gb/s) |
| **PCIe** | Gen 3.0/4.0 x16 | Gen 5.0 x16 |
| **Transistors** | ~10 billion (est.) | 22 billion |
| **Max Power** | ~75 W | ~150 W |
| **Compression Engine** | Deflate only (HW) | Deflate + LZ4 (HW) |
| **Packet Processing** | Software (ARM cores) | APP: 64-128 HW packet cores + DPA: 16 HT cores |
| **RDMA** | RoCE v2, InfiniBand | RoCE v2, InfiniBand |
| **GPUDirect RDMA** | Supported | Supported |
| **Crypto** | AES-XTS, AES-GCM, IPsec | AES-XTS, AES-GCM, IPsec (higher throughput) |
| **Form Factor** | FHHL PCIe card | FHHL PCIe card |
| **Equivalent CPU Cores** | ~125 CPU cores (claimed) | ~300 CPU cores (claimed) |
| **eMMC Flash** | 16-64 GB | 64 GB |
| **OOB Management** | 1 GbE port | 1 GbE port + BMC chip |

### Performance Notes

- BF-2 ARM cores benchmarked roughly comparable to Intel Atom / Celeron N5105 class on Geekbench 5 (not the main value — the HW accelerators are).
- BF-3 Cortex-A78 cores are significantly faster: ~50% higher IPC than A72 + 20% higher clock.
- BF-3 introduces the Accelerated Programmable Pipeline (APP) with 64-128 dedicated packet processing cores that operate on-path, separate from the ARM cores.
- Traversing ARM cores in on-path packet processing adds 1.3x-2x overhead versus pure hardware offload on BF DPUs.

---

## 3. DPU Internal Architecture

```
                    ┌─────────────────────────────────────┐
                    │          BlueField DPU Card          │
                    │                                      │
                    │  ┌──────────┐    ┌───────────────┐  │
                    │  │ ARM SoC  │    │  ConnectX NIC  │  │
                    │  │ 8/16     │    │  (CX-6/CX-7)  │  │
                    │  │ cores    │◄──►│                │  │ ◄──► Network
                    │  │ + DRAM   │    │  RDMA engine   │  │      (100-400G)
                    │  │ + eMMC   │    │  Packet proc.  │  │
                    │  └────┬─────┘    └───────┬────────┘  │
                    │       │                  │           │
                    │  ┌────┴──────────────────┴────────┐  │
                    │  │     Internal PCIe Switch        │  │
                    │  │     (NTB, P2P, bifurcation)     │  │
                    │  └────────────────┬────────────────┘  │
                    └───────────────────┼──────────────────┘
                                        │ PCIe Gen 4/5 x16
                                        ▼
                                   Host PCIe Bus
                                   (CPU, GPU, NVMe)
```

### Key Architectural Points

1. **ARM SoC is a full computer**: Runs Ubuntu/RHEL, has its own DRAM, eMMC storage, can run containers, custom services, full Linux networking stack.

2. **ConnectX NIC is integrated**: Same silicon as standalone ConnectX-6/7 cards. Full RDMA, RoCE, GPUDirect support. The ARM cores can program the NIC's flow tables, steering rules, and DMA engines.

3. **PCIe switch provides three roles**:
   - **Endpoint**: Host sees DPU as a PCIe device (NIC)
   - **Root Complex**: DPU can act as PCIe host (attach NVMe SSDs, GPUs)
   - **Non-Transparent Bridge (NTB)**: DPU can bridge between host PCIe domain and its own

4. **DPU sits on the data path**: All host network traffic flows through the DPU. The DPU decides what reaches the host and what it handles itself.

---

## 4. Memory Architecture

### DPU's Own Memory
- BF-2: 16 or 32 GB DDR4 — private to the ARM cores
- BF-3: 32 GB DDR5 — private to the ARM cores
- This memory is completely separate from host DRAM
- Used for DPU OS, applications, packet buffers, flow tables

### Host Memory Access
- DPU can access host memory via PCIe (DOCA DMA)
- Host exports memory regions (doca_mmap) that DPU can read/write
- This is how the DPU moves data between its own buffers and host pinned memory
- Bandwidth limited by PCIe (Gen4 x16 = ~32 GB/s, Gen5 x16 = ~64 GB/s)

### GPU VRAM Access (Critical for OuterLink)
- The DPU's integrated ConnectX NIC supports GPUDirect RDMA
- ConnectX can read/write GPU VRAM via PCIe BAR1 — same mechanism as OpenDMA
- BF-3 requires `RmDmaAdjustPeerMmioBF3=1` NVIDIA driver option for GPU BAR1 access
- Resizable BAR must be enabled in BIOS
- PCIe topology constraint: GPU and DPU should share the same PCIe root complex for optimal throughput

### Memory Hierarchy (DPU perspective)
```
Fastest ─► DPU on-chip SRAM (packet buffers, flow tables)
           DPU DDR4/DDR5 (ARM core working memory)
           Host DRAM (via PCIe DMA)
           GPU VRAM (via PCIe BAR1 / GPUDirect RDMA)
Slowest ─► Remote GPU VRAM (via network RDMA)
```

---

## 5. PCIe Topology

### Standard Server Configuration
```
                    PCIe Root Complex (CPU)
                    ├── PCIe Slot 1: BlueField DPU
                    │   ├── ARM SoC (endpoint to host)
                    │   ├── ConnectX NIC (network ports)
                    │   └── PCIe Root (can attach downstream devices)
                    ├── PCIe Slot 2: NVIDIA GPU (e.g., RTX 3090)
                    ├── PCIe Slot 3: NVMe SSD
                    └── ...
```

### P2P Considerations
- For GPUDirect RDMA: DPU's ConnectX and GPU should ideally be under the same PCIe root complex
- PCIe switch bifurcation on BF-3 allows x32+x4 configurations
- BF-3 can function as PCIe root complex, allowing direct attachment of downstream devices

### Pedro's Setup Relevance
- Each PC has ConnectX-5 + GPU (3090). Replacing ConnectX-5 with BlueField-2 gives the same ConnectX-6 networking PLUS ARM offload cores.
- The DPU would sit in the same PCIe tree as the GPU, enabling GPUDirect RDMA between DPU's NIC and the local GPU's BAR1.

---

## 6. Host-DPU Communication Paths

| Path | Bandwidth | Latency | Use Case |
|---|---|---|---|
| **PCIe DMA (DOCA DMA)** | ~32 GB/s (Gen4) | ~1-2 us | Bulk data transfer host↔DPU |
| **Virtio (SR-IOV / VFs)** | ~10-25 Gbps | ~5-10 us | Virtual NIC to host VMs |
| **Representors** | Line rate | Hardware | DPU sees host traffic as representor ports |
| **RShim (USB/PCIe)** | ~10 MB/s | High | Debug/management only |
| **OOB 1GbE** | 1 Gbps | Standard | Management network |
| **Shared memory (NTB)** | PCIe speed | ~1 us | Low-latency host↔DPU comms |
| **Comm Channel** | Moderate | ~5-10 us | DOCA control plane messages |

### DOCA Comm Channel
- Purpose-built for host-DPU control plane communication
- Used to exchange descriptors, negotiate memory regions, signal events
- Not for bulk data — use DOCA DMA for that

---

## 7. Availability and Pricing

### BlueField-2 (Used Market)
- Widely available on eBay and refurbished channels due to data center refresh cycles
- Models: 25 GbE (MBF2H332A), 100 GbE (MBF2M516A), 200 GbE (MBF2M345A)
- Dell OEM variants common (e.g., G0DX0)
- Memory options: 16 GB and 32 GB DDR4
- Estimated used pricing: $150-$500 depending on model/speed (check current eBay listings)
- The 25 GbE dual-port models are cheapest, 200 GbE single-port most expensive

### BlueField-3
- Available new from NVIDIA partners
- Significantly more expensive (enterprise pricing, typically $2000+)
- Some Dell OEM units appearing on eBay
- 400 GbE capability, PCIe Gen5

### Recommendation for OuterLink
- **BF-2 for development/testing**: Cheap on used market, same DOCA SDK, sufficient for validating offload design
- **BF-3 for production**: LZ4 hardware compression, more ARM cores, PCIe Gen5, APP packet processing
- Pedro already has ConnectX-5 cards — BF-2 would be a direct upgrade path (same form factor, adds ARM cores)

---

## 8. BlueField-X (DPU + GPU Hybrid)

NVIDIA also offers BlueField-X cards that integrate a datacenter GPU on the same PCB as the DPU. These allow GPU↔NIC communication without crossing the host PCIe bus. While not relevant for Pedro's current setup (separate GPUs), this validates the architectural direction: the industry is converging on tighter GPU-NIC integration, which is exactly what OuterLink's OpenDMA achieves in software.

---

## 9. Verdict for OuterLink

### Why BlueField Is a Natural Fit

1. **DPU has the same ConnectX NIC** OuterLink already targets. Moving from standalone ConnectX-5 to BlueField-2 (ConnectX-6 Dx) is an upgrade, not a redesign.

2. **ARM cores can run OuterLink transport logic** — routing, connection management, page table operations — freeing the host CPU entirely.

3. **DOCA DMA enables DPU-initiated transfers** — the DPU can read/write host memory and (via GPUDirect) GPU VRAM without any host CPU involvement.

4. **Hardware compression** on BF-2 (deflate) and BF-3 (deflate + LZ4) directly offloads R14's compression work.

5. **The DPU processes packets before the host sees them** — latency reduction for routing decisions, ACKs, retransmissions.

6. **Optional by design** — OuterLink must work without a DPU (standard ConnectX NIC). The DPU is a performance accelerator, not a dependency.

### Risks

1. **ARM core performance**: BF-2's A72 cores are modest. Complex page table operations or speculative prefetch scheduling may bottleneck.
2. **PCIe topology**: GPU and DPU must share PCIe root complex for GPUDirect RDMA. Multi-GPU setups need careful slot planning.
3. **DOCA SDK complexity**: C-only API, heavy abstraction layers, NVIDIA ecosystem lock-in for some features.
4. **32 GB DRAM limit**: Large page tables for multi-GPU pools may exceed DPU memory.

---

## Related Documents
- [R14: Transport Compression](../../R14-transport-compression/) — Compression offloaded to DPU hardware
- [R17: Topology-Aware Scheduling](../../R17-topology-aware-scheduling/) — Routing decisions on DPU
- [R10: Memory Hierarchy](../../R10-memory-hierarchy/) — Page tables managed by DPU
- [R11: Speculative Prefetching](../../R11-speculative-prefetch/) — Prefetch logic on DPU ARM cores

## Open Questions
- [ ] What is the actual PCIe topology in Pedro's PCs? Does the GPU share a root complex with the PCIe slot where a BF-2 would go?
- [ ] Can the BF-2's 8 A72 cores handle OuterLink's full transport stack at 100 Gbps line rate, or do we need BF-3?
- [ ] Is 32 GB DPU DRAM sufficient for page tables when managing 24 GB VRAM x multiple GPUs?
- [ ] Does the BF-2's ConnectX-6 Dx support the same BAR1 direct access pattern as Pedro's standalone ConnectX-5 for OpenDMA?
