# R21 Research: Linux P2PDMA Framework and NVMe Direct Access

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Document the Linux kernel P2PDMA framework, NVMe Controller Memory Buffers (CMB), PCIe peer-to-peer DMA between NVMe and NICs, NVMe-oF (NVMe over Fabrics) with RDMA transport, and SPDK for userspace NVMe. These are the building blocks for OuterLink's remote storage-to-GPU pipeline.

---

## 1. Linux Kernel P2PDMA Framework

### Overview

P2PDMA (Peer-to-Peer DMA) is an upstream Linux kernel framework (since 4.20, November 2018) that enables direct DMA transfers between PCIe devices without routing data through system RAM. The data moves directly between device BARs over the PCIe fabric.

### Architecture: Provider, Client, Orchestrator

| Role | Description | Example |
|------|-------------|---------|
| **Provider** | Exposes memory (BAR region) as P2P DMA resource | NVMe with CMB, or GPU exposing BAR1 |
| **Client** | Uses P2P memory for DMA transactions | RDMA NIC, NVMe controller |
| **Orchestrator** | Coordinates data flow between provider and client | NVMe Target driver (nvmet), filesystem |

Roles can overlap. In the NVMe-oF target use case, the NVMe driver acts as all three: it exposes CMB (provider), accepts DMA to CMB (client), and orchestrates the RDMA-to-NVMe flow (orchestrator).

### Kernel Version History

| Version | Capability |
|---------|-----------|
| **4.20** (Nov 2018) | Initial P2PDMA framework merged upstream |
| **5.x** | Available but NOT compiled by default in most distros |
| **6.2** (Feb 2023) | Userspace P2PDMA interface added |
| **6.2+** | NVIDIA GDS can use kernel P2PDMA instead of nvidia-fs.ko |
| **Ubuntu 24.04** | P2PDMA enabled by default (CONFIG_PCI_P2PDMA=y) |

### Kernel Configuration

```
CONFIG_PCI_P2PDMA=y   (under Device Drivers > PCI support)
```

Most modern distributions enable this. Ubuntu 24.04 has it on by default.

### Key API Functions

| Function | Purpose |
|----------|---------|
| `pci_p2pdma_add_resource()` | Provider registers a BAR region as P2P memory |
| `pci_p2pmem_publish()` | Provider makes P2P memory discoverable |
| `pci_p2pdma_distance()` | Check if P2P is supported between two devices |
| `pci_alloc_p2pmem()` | Allocate memory from a P2P provider |
| `pci_p2pdma_map_sg()` | Map P2P memory for DMA by a client device |

### PCIe Topology Requirements

This is the most critical constraint for P2PDMA:

| Topology | P2P Support | Notes |
|----------|-------------|-------|
| **Same PCIe switch** | Always works | TLP routing stays within switch hierarchy |
| **Same root complex (AMD Zen)** | Works | AMD root complex supports peer forwarding |
| **Same root complex (Intel)** | Whitelisted only | Kernel maintains a whitelist of known-good Intel chipsets |
| **Different root complexes** | Blocked by default | PCIe spec doesn't guarantee forwarding across root complexes |
| **CPU root port hairpin** | May work | Some CPUs route P2P through CPU, doubling latency |

**For OuterLink's setup:** AMD Zen chipsets (Ryzen, EPYC) have excellent P2PDMA support for both reads and writes. All our PCIe devices behind the same root complex can do P2P.

### Performance Impact

| Metric | Improvement with P2PDMA |
|--------|------------------------|
| CPU memory load | ~50x reduction |
| CPU PCIe bus load | ~25x reduction |
| CPU core utilization | ~50% reduction |
| Latency | Lower (single DMA hop vs two) |

---

## 2. NVMe Controller Memory Buffer (CMB)

### What CMB Is

The NVMe spec (since 2014) allows an SSD to expose on-chip SRAM via a PCIe BAR — the Controller Memory Buffer. This memory can be used as a staging area for P2P DMA operations.

### CMB for P2PDMA

| Feature | Detail |
|---------|--------|
| **Size** | Typically 4MB-256MB (controller-dependent) |
| **Access** | Exposed as PCIe BAR, mappable by any PCIe device |
| **P2PDMA role** | Acts as Provider — other devices DMA to/from CMB |
| **NVMe driver support** | `nvme` driver registers CMB as P2P resource automatically |
| **Availability** | Rare in consumer NVMe. More common in enterprise / datacenter SSDs |

### CMB Limitations

- **Consumer NVMe SSDs generally lack CMB** — Samsung 990 Pro, WD SN850X, etc. do not have CMB
- CMB is small (megabytes, not gigabytes) — only useful as a staging buffer, not for entire datasets
- Without CMB, the NVMe device cannot be a P2PDMA provider (but can still be a client)

### CMB Alternatives

When NVMe lacks CMB, two alternatives exist:

1. **RNIC BAR as provider**: If the NIC (e.g., ConnectX-5) exposes BAR memory, the NVMe target can use the NIC's memory as the P2P staging area instead of CMB
2. **Host-staged with pinned memory**: Fall back to DMA through pinned system RAM (this is what we do in Phase 1 transport anyway)

---

## 3. NVMe-to-NIC Direct DMA (P2P Storage Networking)

### The Ideal Path

```
NVMe SSD --P2P DMA--> ConnectX-5 NIC --RDMA--> wire
```

This is the sender-side of our remote GDS pipeline. Can the NVMe controller DMA data directly to the NIC for network transmission, without touching system RAM?

### ConnectX-5 Embedded PCIe Switch

This is a key enabler. ConnectX-5 contains an embedded PCIe switch that:

- Creates a shared PCIe hierarchy between the NIC ports and any devices behind the switch
- Enables P2P DMA between the NIC and locally connected NVMe drives
- Used by NVMe-oF target offload to serve NVMe data over the network with zero CPU involvement

| Feature | ConnectX-5 Capability |
|---------|----------------------|
| Embedded PCIe switch | Yes |
| NVMe-oF target offload | Yes (hardware-accelerated) |
| P2P with local NVMe | Yes (via embedded switch) |
| CPU involvement | Zero for data path |
| Storage protocols | NVMe-oF, iSER, SRP, NFS RDMA, SMB Direct |

### How NVMe-oF Target Offload Works

1. Remote initiator sends NVMe-oF read command over RDMA
2. ConnectX-5 hardware processes the NVMe-oF command (no CPU)
3. ConnectX-5 DMA engine reads data from local NVMe via embedded PCIe switch (P2P)
4. ConnectX-5 sends data over RDMA to the remote initiator
5. CPU never touches the data

### Practical Constraint: PCIe Topology

For NVMe-to-NIC P2P to work, devices must share a PCIe hierarchy:

| Configuration | Works? | Notes |
|---------------|--------|-------|
| NVMe behind ConnectX-5 embedded switch | Yes (best) | Guaranteed P2P routing |
| NVMe and ConnectX-5 on same PCIe switch | Yes | External switch also works |
| NVMe and ConnectX-5 on same AMD root complex | Yes | AMD Zen supports this |
| NVMe and ConnectX-5 on different root complexes | Unlikely | Kernel blocks by default |
| NVMe in M.2 slot, ConnectX-5 in PCIe slot (typical desktop) | Depends on chipset | AMD Zen: likely works. Intel: check whitelist |

### Real-World Example: Chelsio T7

Chelsio's T7 NIC demonstrates NVMe/TCP PDU offload with CMB-based P2P: data moves directly between NVMe CMB and the NIC without host memory or CPU involvement. This validates the architecture even if the specific hardware differs.

---

## 4. NVMe over Fabrics (NVMe-oF)

### Overview

NVMe-oF extends the NVMe protocol over a network fabric, allowing remote access to NVMe drives with near-local performance. It is the standard protocol for remote NVMe access.

### Transport Options

| Transport | Latency (added) | Throughput | CPU Overhead | Notes |
|-----------|-----------------|-----------|--------------|-------|
| **RDMA (RoCEv2)** | 5-10 us | Line rate (100 Gbps) | Very low | Best for OuterLink — ConnectX-5 native |
| **RDMA (InfiniBand)** | 2-5 us | Line rate | Very low | Even lower latency, requires IB fabric |
| **TCP** | 30-100 us | High but CPU-bound | High | Fallback when RDMA unavailable |
| **FC (Fibre Channel)** | 5-10 us | FC speed-dependent | Low | Enterprise SAN, not relevant for us |

### NVMe-oF with RDMA — Performance

| Metric | Value | Source |
|--------|-------|--------|
| Added latency (RDMA) | Sub-10 microseconds | Industry benchmarks |
| Aggregate bandwidth | 100 Gbps line rate (~12.5 GB/s) | ConnectX-5 100GbE |
| IOPS | >1 million | With multiple NVMe drives |
| CPU utilization | Minimal with RDMA | Hardware offload handles data path |

### NVMe-oF Architecture for OuterLink

On the **storage node** (sender):
```
Local NVMe --> nvmet (kernel NVMe target) --> RDMA transport --> ConnectX-5 --> wire
```

On the **GPU node** (receiver):
```
wire --> ConnectX-5 --> RDMA transport --> nvme-cli (initiator) --> appears as /dev/nvmeXnY
```

With NVMe-oF target offload on ConnectX-5, the storage node's data path is entirely in hardware.

### NVMe-oF vs Direct RDMA for OuterLink

| Approach | Pros | Cons |
|----------|------|------|
| **NVMe-oF** | Standard protocol, hardware offloads, block device semantics | Protocol overhead, requires nvmet setup, block-level only |
| **Custom RDMA** | Full control, can integrate with memory tiering, DMA directly to GPU BAR1 | Must build protocol, no hardware NVMe-oF offloads |
| **Hybrid** | Use NVMe-oF for bulk reads, custom RDMA for GPU delivery | Two paths to manage |

---

## 5. SPDK (Storage Performance Development Kit)

### Overview

SPDK is Intel's open-source userspace storage framework. It provides:

- Userspace NVMe driver (bypasses kernel entirely)
- Polled I/O model (no interrupts, no context switches)
- NVMe-oF target and initiator in userspace
- RDMA transport support (libibverbs / rdmacm)
- Zero-copy data path from network to storage

### Architecture

```
Application
    |
SPDK Libraries (userspace)
    |
    +-- NVMe driver (UIO/VFIO)     -- direct NVMe access
    +-- NVMe-oF target              -- serve NVMe over network
    +-- RDMA transport              -- libibverbs/rdmacm
    +-- Blobstore / BlobFS          -- lightweight filesystem
```

### SPDK vs Kernel NVMe Stack

| Aspect | Kernel NVMe | SPDK |
|--------|-------------|------|
| **Latency** | ~10 us (interrupt-driven) | ~2-5 us (polled) |
| **CPU model** | Interrupt-based, shared | Dedicated cores, polled |
| **Throughput** | Good | Better (no context switches) |
| **Kernel dependency** | Full kernel stack | UIO/VFIO only |
| **Filesystem support** | ext4, XFS, etc. | BlobFS (limited) or raw blocks |
| **P2PDMA integration** | Native (kernel framework) | SPDK has its own P2P support |
| **Complexity** | Standard Linux | Dedicated cores, custom setup |

### SPDK P2P DMA Support

SPDK includes P2P DMA support that works with NVMe CMBs:

- SPDK can allocate I/O buffers from P2P memory (CMB or other BAR regions)
- NVMe-oF target can use P2P memory for zero-copy transfers
- Works with ConnectX-5 RDMA transport

### SPDK NVMe-oF RDMA Performance

| Configuration | Metric | Value |
|---------------|--------|-------|
| ConnectX-5 Ex, RoCEv2, SPDK 24.05 | Throughput | Near line rate |
| ConnectX-5 Ex, RoCEv2 | IOPS (4K random) | >1M with multiple drives |
| CPU offload via target offload | CPU reduction | ~38.7% less utilization |

### SPDK Relevance for OuterLink

| Use Case | Applicable? | Notes |
|----------|-------------|-------|
| Storage node NVMe access | Possible but heavy | Requires dedicated cores, VFIO setup |
| NVMe-oF target | Yes — better latency than kernel | But kernel nvmet with ConnectX-5 offload may be simpler |
| RDMA transport | Yes — but we already have our own | SPDK RDMA is libibverbs-based like ours |
| P2P with GPU | Not directly | SPDK doesn't know about GPU BAR1 |

**Verdict:** SPDK is powerful but introduces significant complexity (dedicated cores, VFIO, BlobFS). For OuterLink, the kernel NVMe-oF stack with ConnectX-5 hardware offload is simpler and achieves similar throughput. SPDK becomes interesting only if we need sub-5us storage latency or want to bypass the kernel entirely on the storage node.

---

## 6. Realistic P2P Paths for OuterLink

### Path 1: NVMe --> NIC (Sender Side)

```
NVMe SSD --(PCIe P2P)--> ConnectX-5 embedded switch --(RDMA)--> wire
```

**Requirements:**
- ConnectX-5 with NVMe-oF target offload configured
- NVMe drive accessible via nvmet subsystem
- PCIe topology allows P2P (AMD Zen or shared switch)

**Performance:** Limited by NVMe read speed (~7 GB/s per Gen4 drive)

### Path 2: NIC --> GPU (Receiver Side — OpenDMA)

```
wire --(RDMA)--> ConnectX-5 DMA engine --(PCIe BAR1)--> GPU VRAM
```

**Requirements:**
- ConnectX-5 with RDMA configured
- GPU BAR1 mapped and accessible (OpenDMA)
- PCIe topology allows DMA to BAR1

**Performance:** Limited by network speed (~12.5 GB/s for 100GbE)

### Path 3: NVMe --> Host RAM --> NIC (Fallback Sender)

```
NVMe SSD --(DMA)--> pinned host RAM --(RDMA)--> ConnectX-5 --> wire
```

**Requirements:** None special — standard NVMe read + RDMA send
**Performance:** Same throughput, but uses system RAM bandwidth and CPU

### Path 4: NIC --> Host RAM --> GPU (Fallback Receiver)

```
wire --> ConnectX-5 --(RDMA)--> pinned host RAM --(cudaMemcpy)--> GPU VRAM
```

**Requirements:** None special — this is Phase 1 transport
**Performance:** Uses system RAM bandwidth, adds latency

---

## 7. Key Takeaways

1. **P2PDMA is mature and upstream** — kernel 6.2+ with userspace support, enabled by default in Ubuntu 24.04. AMD Zen has full support.

2. **NVMe CMB is rare in consumer SSDs** — cannot rely on it. Must support non-CMB paths (host-staged or NIC BAR as P2P provider).

3. **ConnectX-5 embedded PCIe switch is the enabler** — allows NVMe-to-NIC P2P without depending on NVMe CMB. Hardware NVMe-oF target offload handles the complete data path.

4. **NVMe-oF with RDMA is the standard remote storage protocol** — sub-10us added latency, line-rate throughput, hardware offloaded on ConnectX-5.

5. **SPDK adds performance but also complexity** — kernel NVMe-oF with ConnectX-5 offload is sufficient for our throughput needs. SPDK is a potential optimization for later.

6. **The bottleneck is always the NVMe SSD** — ~7 GB/s per Gen4 drive. Network (12.5 GB/s) and GPU BAR1 have headroom. Stripe across multiple drives to scale.

7. **AMD Zen chipsets are ideal** — full P2PDMA support for reads and writes across the root complex.

---

## Related Documents

- [01-nvidia-gds-architecture.md](01-nvidia-gds-architecture.md) — NVIDIA's proprietary GDS approach
- [03-remote-gds-pipeline.md](03-remote-gds-pipeline.md) — Combining P2PDMA + RDMA for remote storage
- [R10: Memory Tiering](../../../../phase-08-smart-memory/R10-memory-tiering/README.md) — NVMe as Tier 4/5

## Open Questions

- [ ] Do our consumer NVMe SSDs (Samsung 990 Pro, etc.) have CMB? (Likely not — need to verify with `nvme id-ctrl /dev/nvme0 | grep cmb`)
- [ ] Can ConnectX-5 NVMe-oF target offload work with M.2 NVMe drives on the motherboard? (Depends on whether the M.2 slot is behind the CPU root complex or a chipset switch)
- [ ] Does SPDK's userspace NVMe driver provide better P2P performance than the kernel nvme driver for our use case?
- [ ] What's the actual P2PDMA bandwidth between M.2 NVMe and ConnectX-5 PCIe slot on our AMD Zen boards?
- [x] Is P2PDMA upstream and production-ready? **Yes — kernel 6.2+ with userspace support, enabled by default in Ubuntu 24.04**
