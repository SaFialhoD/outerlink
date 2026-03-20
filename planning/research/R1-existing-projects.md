# R1: Existing GPU Sharing / Remoting / Virtualization Projects

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** HIGH

## Purpose

Map the entire landscape of GPU sharing, remoting, and virtualization projects to understand what exists, what works, what failed, and where the gap is.

## TL;DR - THE GAP IS REAL

There is **no modern, open-source, production-grade solution** for remote GPU access over standard Ethernet with current CUDA support. The closest projects:

| Project | Why It's Close | Why It's Not Enough |
|---------|---------------|-------------------|
| **SCUDA** | LD_PRELOAD, TCP, open source, active | Very early stage, TCP-only, limited API |
| **gVirtuS** | CUDA 12.6+, GPL, has TCP+RDMA | Performance issues on small kernels |
| **Cricket** | Driver API, open source (GPLv3), active | Focused on checkpoint/restart, not pooling |
| **TensorFusion** | Most polished, Kubernetes-native, <4% overhead | Advanced features partially closed source |

**OutterLink fills a clear market gap.**

---

## 1. Remote GPU Projects (Most Relevant to Us)

### SCUDA - GPU over IP (CLOSEST TO OUTTERLINK)
- **URL:** github.com/kevmo314/scuda
- **Approach:** LD_PRELOAD intercepts CUDA calls, code-generated RPC stubs from CUDA headers, TCP transport (port 14833)
- **API Coverage:** cuda.h, cublas_api.h, cudnn_graph.h, cudnn_ops.h
- **Status:** Active, early stage, significant GitHub/HN interest
- **Performance:** TCP adds significant latency - acknowledged as dev/test only, not production
- **License:** Open source
- **Relevance:** Nearly identical concept to OutterLink but TCP-only and early. Study their code-generation approach for RPC stubs.

### gVirtuS - GPU Virtualization Service
- **URL:** github.com/gvirtus/GVirtuS
- **Approach:** Split-driver model (frontend in VM, backend on GPU host). Multiple communicator plugins: TCP/IP, RDMA, VMSocket, shared memory
- **API Coverage:** CUDA Runtime, CUDA Driver, OpenCL
- **CUDA Support:** 12.6+ (most current of any open source project)
- **Status:** Active, community maintained, ARM support
- **Performance:** Zero overhead on large kernels, 1.05-1.45x on small kernels
- **License:** GPL
- **Relevance:** Most mature open source option. Has RDMA transport plugin. Study their communicator abstraction layer.

### Cricket - GPU Virtualization with Checkpoint/Restart
- **URL:** github.com/RWTH-ACS/cricket
- **Approach:** Driver API interception with checkpoint/restart support. Can migrate GPU state between GPUs.
- **Status:** Active, open source (GPLv3)
- **Unique:** GPU state checkpoint and live migration
- **Relevance:** Only open source project doing Driver API interception for remoting. Study their handle translation and state management.

### rCUDA - Remote CUDA (Pioneer)
- **URL:** rcuda.net
- **Approach:** Client-server, Runtime API wrappers, TCP or InfiniBand transport
- **Status:** Stalled at CUDA 9.0 (last release July 2020)
- **Performance:** Near-native over InfiniBand. Published: 2x cluster throughput, 40% energy reduction.
- **License:** Proprietary/academic (binary only)
- **Relevance:** Most cited and researched. Good for understanding the problem space. Cannot use code.

### vCUDA
- **Status:** Dead. Published 2009. CUDA 2.x support only.
- **Relevance:** Historical. Introduced "lazy updates" (call batching) concept which is still valuable.

### qCUDA
- **URL:** github.com/coldfunction/qCUDA
- **Approach:** CUDA Runtime interception via virtio (QEMU/KVM)
- **Status:** Research prototype, 32 CUDA APIs only
- **Relevance:** Limited, but shows virtio approach for VM environments.

### AVEC - Accelerator Virtualization in Cloud-Edge
- **Approach:** API interception over TCP/IP, designed for IoT/edge devices without GPUs
- **Status:** Research paper (2023)
- **Relevance:** Validates TCP/IP approach for edge scenarios.

---

## 2. Commercial GPU Remoting (Our Competition)

### VMware Bitfusion
- **Approach:** CUDA interception, network transport (auto-selects: CPU copies, PCIe, Ethernet, IB, RDMA)
- **Requirements:** 10 Gbps+ bandwidth, <50us latency for multi-GPU
- **Status:** Production but uncertain future (Broadcom acquisition)
- **Relevance:** Proves the concept works commercially. Their auto-transport selection is smart.

### VirtAI OrionX
- **Approach:** CUDA interception, 25G RDMA transport, supports splitting + aggregation + remote + live migration
- **Performance:** Claims >95% of local GPU on 25G RDMA
- **Status:** Active commercial (China-focused)
- **Relevance:** Closest commercial analog to OutterLink vision. Shows what's achievable with RDMA.

### TensorFusion (Open Core)
- **URL:** github.com/NexusGPU/tensor-fusion
- **Approach:** Rust/C++ virtualization, Kubernetes-native, GPU-over-IP
- **Performance:** Claims <4% overhead, sometimes exceeds direct GPU access
- **Requirements:** NVIDIA Volta+, driver 530+, CUDA 12.1+
- **License:** Apache 2.0 core, advanced features partially closed
- **Relevance:** Most polished modern implementation. Study their architecture. Free for <10 GPUs.

### Run:ai
- **Status:** Acquired by NVIDIA (2024). Production at 50,000+ GPU scale.
- **Relevance:** Validates market. Now NVIDIA-owned, so unlikely to be open.

---

## 3. GPU Partitioning / Sharing (Local Only - Context)

### NVIDIA MPS (Multi-Process Service)
- Software SM partitioning. Multiple processes share one GPU.
- Up to 3.5x throughput for small workloads.
- No memory bandwidth isolation. No error isolation.

### NVIDIA MIG (Multi-Instance GPU)
- Hardware partitioning into up to 7 isolated instances.
- Ampere+ only (A100, H100, etc.). **NOT available on RTX 3090.**
- Full QoS and fault isolation.

### NVIDIA vGPU / GRID
- Hypervisor-level GPU sharing for VDI.
- Requires commercial license. Up to 32 virtual desktops per GPU.

### HAMi (CNCF Sandbox)
- **URL:** github.com/Project-HAMi/HAMi
- CUDA API interception for memory/compute quotas in Kubernetes.
- 10,000+ concurrent Pods in production. Improved utilization 13% -> 37%.
- **HAMi-core** is the gold standard for CUDA interception (222 functions).

### Fractional GPUs (FGPU)
- **URL:** github.com/sakjain92/Fractional-GPUs
- Software memory bandwidth isolation via reverse-engineered page coloring.
- Only project to solve memory bandwidth isolation in software.
- Academic, 2019, requires GPU memory hierarchy knowledge.

---

## 4. Hardware Approaches (Future Context)

### PCIe Fabric Solutions
| Product | Approach | Latency | Status |
|---------|----------|---------|--------|
| **GigaIO FabreX** | PCIe fabric switching | <130ns | Active commercial |
| **Liqid** | PCIe Gen5 + CXL 2.0 | Near-native | Active, $52M+ DoD contracts |
| **DxPU (Alibaba)** | Custom PCIe TLP-to-network | 4.9-6.8us | Production at scale |
| **SmartIO** | Native PCIe over NTB | Zero penalty | Academic |

### CXL (Compute Express Link)
- Cache-coherent interconnect on PCIe physical layer
- 3.8x speedup vs 200G RDMA for LLM inference memory sharing
- CXL 2.0 in production, 3.0 spec available
- **This is the future of memory disaggregation** - but hardware is still emerging

### Thunderbolt eGPU
- 40 Gbps (TB3/4), 80 Gbps (TB5)
- 1:1 only, no sharing, significant bandwidth bottleneck vs native PCIe
- Not relevant for our use case

---

## 5. Key Academic Papers

| Paper | Year | Key Finding |
|-------|------|-------------|
| GPU Virtualization Survey | 2017 | Comprehensive taxonomy of approaches |
| Fractional GPUs | 2019 | Software memory bandwidth isolation via page coloring |
| GPUPool | 2022 | QoS-guaranteed fine-grained sharing |
| DxPU | 2023 | Production GPU disaggregation at PCIe level |
| G-Safe | 2024 | 4-12% overhead for CUDA monitoring via LD_PRELOAD |
| Interception Inception | 2024 | Undocumented "export tables" for hidden CUDA call interception |
| Prism (Meta) | 2025 | Production DLRM disaggregation |
| Disaggregated Memory Survey | 2025 | Comprehensive cross-layer survey |

---

## 6. Competitive Analysis Matrix

| | OutterLink (planned) | SCUDA | gVirtuS | Cricket | TensorFusion | Bitfusion | OrionX |
|---|---|---|---|---|---|---|---|
| **Open Source** | Yes | Yes | Yes (GPL) | Yes (GPLv3) | Partial | No | No |
| **Interception Level** | Driver API | Runtime | Runtime+Driver | Driver | ? | ? | ? |
| **Transport** | TCP (upgradable) | TCP | TCP/RDMA/VMSocket | Yes | Network | Auto-select | 25G RDMA |
| **CUDA Version** | Modern | Modern | 12.6+ | Modern | 12.1+ | ? | Modern |
| **GPU Pooling** | Planned | Planned | No | No | Yes | Yes | Yes |
| **Memory Sharing** | Planned | No | No | No | Yes | Yes | Yes |
| **Multi-node** | Planned | No | Yes | Yes | Yes | Yes | Yes |
| **Status** | Pre-planning | Early | Active | Active | Active | Uncertain | Active |

---

## 7. Conclusions & Impact on OutterLink

### What to Build On
1. **HAMi-core's interception pattern** - 222 functions, version-indexed, `cuGetProcAddress` hooks
2. **gVirtuS's communicator abstraction** - pluggable transport layer
3. **SCUDA's code-gen approach** - auto-generate RPC stubs from CUDA headers
4. **vCUDA's lazy updates** - batch non-side-effect calls

### What to Build Different
1. **Transport:** Start TCP + io_uring, design for RDMA upgrade (not SoftRoCE)
2. **Interception:** Driver API (not Runtime API like rCUDA/gVirtuS)
3. **Scope:** GPU + system RAM pooling (not just GPU sharing)
4. **Target:** Commodity hardware (not Kubernetes-only, not datacenter-only)

### What NOT to Build (Solved Problems)
1. Don't reinvent CUDA interception mechanics - use HAMi-core patterns
2. Don't build Kubernetes integration for v1 - that's a later feature
3. Don't build hardware partitioning - that's NVIDIA MIG territory

### Projects to Study Deeply
1. **SCUDA** (github.com/kevmo314/scuda) - closest concept, study RPC generation
2. **Cricket** (github.com/RWTH-ACS/cricket) - Driver API remoting, study handle translation
3. **HAMi-core** (github.com/Project-HAMi/HAMi-core) - interception mechanics
4. **gVirtuS** (github.com/gvirtus/GVirtuS) - transport abstraction, CUDA 12.6 support

## Related Documents

- [Project Vision](../../docs/architecture/00-project-vision.md)
- [R2: SoftRoCE Research](R2-softroce-rdma.md)
- [R3: CUDA Interception](R3-cuda-interception.md)
- [Pre-Planning Master](../pre-planning/00-master-preplan.md)

## Open Questions

- [ ] Should we fork SCUDA or Cricket as a starting point, or build from scratch?
- [ ] What license should OutterLink use? (GPL like gVirtuS/Cricket, or Apache 2.0 like HAMi?)
- [ ] Should we target compatibility with TensorFusion's Kubernetes integration later?
- [ ] Is the "Interception Inception" export tables technique documented enough to use?
