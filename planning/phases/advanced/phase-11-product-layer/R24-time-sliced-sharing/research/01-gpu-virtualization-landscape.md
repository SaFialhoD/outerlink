# R24 Research: GPU Virtualization & Sharing Landscape

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Map every existing GPU sharing and virtualization technology, understand what works on GeForce consumer GPUs, and identify what OuterLink must build vs. what it can leverage.

---

## TL;DR

| Technology | Works on GeForce? | Isolation | Sharing Granularity | Open Source? |
|---|---|---|---|---|
| NVIDIA MPS | Yes (CC 3.5+) | Weak (pre-Volta) / Moderate (Volta+) | Concurrent kernels | N/A (NVIDIA binary) |
| NVIDIA MIG | **No** (datacenter only) | Strong (HW partitioned) | Fixed GPU slices (up to 7) | N/A (NVIDIA binary) |
| NVIDIA vGPU | **No** (licensed, datacenter) | Strong (hypervisor) | Virtual GPU profiles | Proprietary |
| K8s Time-Slicing | Yes | None (round-robin) | Oversubscribed replicas | Yes (GPU Operator) |
| HAMi | Yes | Soft (API interception) | Memory + compute quotas | Yes (Apache 2.0, CNCF) |
| gVisor | Yes (with caveats) | Moderate (syscall proxy) | Single GPU per sandbox | Yes (Apache 2.0) |
| Kata Containers | Yes (passthrough) | Strong (VM) | Single GPU per VM | Yes (Apache 2.0) |

**Key insight:** On GeForce hardware, MPS and software-level interception (HAMi's approach) are the only viable sharing mechanisms. MIG and vGPU are locked to datacenter GPUs. OuterLink already does CUDA interception, so we can build sharing at the same layer HAMi does, but across the network.

---

## 1. NVIDIA MPS (Multi-Process Service)

### What It Does

MPS replaces the default CUDA context-switching model with a shared hardware context. Multiple CUDA processes submit work to the same GPU concurrently via Hyper-Q, avoiding the overhead of full context switches.

### How It Works

- MPS daemon runs as a background service, creating a single server process per GPU
- Client CUDA processes connect to the MPS server instead of creating individual GPU contexts
- Kernels from different clients run concurrently on the same SMs (Streaming Multiprocessors)
- CUDA streams from different clients can overlap

### Limitations

| Limitation | Details |
|---|---|
| Client limit | Pre-Volta: 16 clients. Volta+: 48-60 clients per GPU |
| Memory isolation | Pre-Volta: **none** (shared address space, out-of-range writes corrupt other processes). Volta+: fully isolated GPU address spaces |
| Error containment | Pre-Volta: fatal fault kills ALL clients. Volta+: contained to subset of shared GPUs |
| Single user | Only one OS user can have an active MPS server at a time |
| Client termination | Terminating a client without GPU sync can leave MPS server in undefined state |
| No VRAM quotas | MPS does not enforce memory limits per client |
| No priority scheduling | All clients get equal access, no preemption |

### GeForce Compatibility

MPS works on any GPU with compute capability 3.5+ (Kepler and later). GeForce GTX 780 and newer qualify. However, GeForce GPUs lack Exclusive Process compute mode, and Hyper-Q support may be more limited than on datacenter GPUs. Our RTX 3090s (Ampere, CC 8.6) have full Volta+ MPS with address space isolation.

### Relevance to OuterLink

MPS could be used as a LOCAL sharing mechanism on each GPU node. OuterLink intercepts CUDA calls before they reach the GPU, so we can route multiple remote users' calls through MPS on the target GPU. However, MPS provides no memory quotas or priority scheduling, so OuterLink must implement those in its scheduling layer.

---

## 2. NVIDIA MIG (Multi-Instance GPU)

### What It Does

MIG hardware-partitions a single GPU into up to 7 isolated instances, each with dedicated SMs, L2 cache, and memory controllers. Each instance acts like a separate GPU.

### Supported GPUs

- A30, A100 (Ampere)
- H100, H200 (Hopper)
- B200 (Blackwell)
- RTX PRO 6000 (Blackwell, with graphics support)
- **NOT supported on ANY GeForce GPU** (not even RTX 4090 or 5090)

### Key Properties

| Property | Details |
|---|---|
| Isolation | Full HW isolation: separate SMs, L2 cache, memory bandwidth, error containment |
| P2P | Only between MIG instances on same GPU (not cross-GPU) |
| NVLink | Completely unavailable in MIG mode |
| NCCL | Not supported with MIG |
| CUDA MPS | Can run MPS on top of MIG instances |
| Reconfiguration | Requires GPU reset on Ampere, no reset needed on Hopper+ |

### Relevance to OuterLink

**Not applicable for our GeForce hardware.** However, for datacenter deployments of OuterLink, MIG could partition GPUs before OuterLink pools them. A single A100 could appear as 7 separate GPU resources in the OuterLink pool. This is a future consideration for enterprise users.

---

## 3. NVIDIA vGPU

### What It Does

NVIDIA vGPU (Virtual GPU) technology enables multiple virtual machines to share a single physical GPU, with each VM getting a virtual GPU instance. Runs on top of hypervisors (VMware vSphere, Citrix, Red Hat KVM).

### Licensing

- Requires NVIDIA AI Enterprise or vGPU software license (paid subscription)
- License types: vPC (virtual PC), vWS (virtual workstation), vApps, C-series (compute)
- Enforced through software: after 20 minutes without license, performance degrades
- C-series (compute) requires NVIDIA AI Enterprise, not available with basic vGPU software

### Supported Hypervisors

VMware vSphere, Citrix XenServer, Red Hat Enterprise Linux KVM, Nutanix AHV, SLES, Proxmox VE (generic KVM). Windows Server 2025 Hyper-V adds GPU Partitioning (GPU-P).

### Key Limitations

| Limitation | Details |
|---|---|
| GeForce | **Not supported at all** |
| Cost | Per-GPU subscription licensing |
| Wayland | Not supported (Red Hat 10.0/10.1 CLI-only) |
| UVM | Disabled by default, must enable per-VM |
| Performance | Lower than bare-metal passthrough |
| Multi-GPU | Complex configuration per VM |

### Relevance to OuterLink

**Not applicable.** vGPU requires datacenter GPUs with paid licensing. OuterLink's software-level interception achieves similar sharing without any NVIDIA licensing. We ARE the alternative to vGPU for people with GeForce hardware.

---

## 4. Kubernetes GPU Time-Slicing (NVIDIA GPU Operator)

### What It Does

The NVIDIA GPU Operator's time-slicing feature oversubscribes GPUs by creating multiple "replicas" of a single GPU device. Each replica is a logical device that Kubernetes can assign to a pod. The GPU time-slices between pods using standard CUDA context switching.

### How It Works

- Administrator defines a `ClusterPolicy` with `devicePlugin.config` specifying replica count
- GPU Operator advertises N replicas of each GPU to the Kubernetes scheduler
- Pods requesting `nvidia.com/gpu: 1` get access to a time-sliced replica
- GPU hardware round-robins between active CUDA contexts

### Key Properties

| Property | Details |
|---|---|
| Isolation | **None** — no memory or fault isolation between replicas |
| Memory | No per-pod VRAM limits; any pod can consume all VRAM |
| Compute | No guaranteed compute share; round-robin with no priority |
| GPU support | Works on any NVIDIA GPU (including GeForce) |
| Configuration | Via ConfigMap + node labels, no GPU restart needed |

### Relevance to OuterLink

Kubernetes time-slicing is the simplest form of GPU sharing and shows the baseline user expectation. OuterLink should provide BETTER sharing than this: actual VRAM quotas, priority levels, and fair-share scheduling. We also extend this across the network, not just within a single node.

---

## 5. HAMi (Heterogeneous AI Computing Virtualization Platform)

### What It Does

HAMi is a CNCF Sandbox project (formerly k8s-vGPU-scheduler) that provides GPU sharing on Kubernetes with actual resource isolation. It intercepts CUDA API calls to enforce memory and compute quotas per container.

### How It Works

- Uses CUDA API interception (similar concept to OuterLink's LD_PRELOAD)
- Tracks memory allocations per container, enforces VRAM limits
- Monitors compute usage, enforces core quotas over time windows
- Kubernetes-native: works with standard pod specs, adds resource annotations

### Key Features

| Feature | Details |
|---|---|
| Memory quotas | Allocate specific VRAM amount (e.g., 3000MB) or percentage (e.g., 50%) |
| Compute quotas | Core usage limits enforced over time windows |
| Device splitting | `deviceSplitCount: 10` means up to 10 pods per GPU |
| Multi-device | Supports GPU, NPU, MLU (Cambricon), and other accelerators |
| Scheduling | Binpack, spread, and topology-aware placement policies |
| Monitoring | Unified metrics across device vendors |

### Relevance to OuterLink

**HAMi is the closest existing project to what R24 wants to build.** Key differences:
- HAMi is local (same-node GPU sharing). OuterLink extends this across the network.
- HAMi is Kubernetes-only. OuterLink should work standalone and with Kubernetes.
- HAMi's interception is at the same CUDA API layer OuterLink already intercepts.
- We can study HAMi's quota enforcement approach and adapt it for our distributed scheduler.

HAMi is Apache 2.0 licensed, so we can reference their approach and potentially integrate with their Kubernetes scheduling.

---

## 6. gVisor GPU Support

### What It Does

gVisor is a container runtime sandbox from Google that intercepts all system calls. For GPU workloads, it proxies NVIDIA driver calls through its security boundary.

### How It Works

- gVisor's Sentry process intercepts all syscalls from the container
- GPU access is proxied through gVisor's NVIDIA driver interception layer
- Supports CUDA, PyTorch, LLM inference, Vulkan, and NVENC/NVDEC

### Key Limitations

| Limitation | Details |
|---|---|
| NVIDIA ABI dependency | Must update gVisor for each new NVIDIA driver version |
| No GPU sharing | One GPU per sandbox, no multiplexing |
| Overhead | Syscall interception adds latency |
| Maintenance burden | Tied to Google's update cycle for driver ABI support |

### Relevance to OuterLink

gVisor's approach is interesting for security isolation but not for GPU sharing. If OuterLink users want security isolation, they'd use gVisor or Kata as the container runtime, with OuterLink handling the GPU pooling layer above.

---

## 7. Kata Containers GPU Passthrough

### What It Does

Kata Containers runs each container in a lightweight VM (QEMU). GPU access is via VFIO/PCI passthrough, giving the VM direct hardware access to the GPU.

### Current State

| Property | Details |
|---|---|
| GPU access | PCI passthrough via VFIO (one GPU per VM) |
| Multi-GPU | Not supported (single GPU passthrough only; Blackwell may add multi-GPU) |
| vGPU | Not supported |
| Isolation | Strong (full VM isolation) |
| NVIDIA support | Technology Preview, NVIDIA GPU Operator integration exists but fragile |
| Prerequisites | IOMMU, hardware virtualization, custom guest kernel for GPU support |

### Relevance to OuterLink

Kata provides the strongest isolation model but the worst sharing model (one GPU per VM). OuterLink could run INSIDE a Kata VM as a client, connecting to remote GPU servers. This gives Kata-level isolation for the application while OuterLink provides GPU access across the network.

---

## Summary: What Works on GeForce

| Approach | Works? | Sharing? | Isolation? |
|---|---|---|---|
| MPS | Yes | Concurrent execution | Volta+ only |
| MIG | **No** | N/A | N/A |
| vGPU | **No** | N/A | N/A |
| K8s Time-Slicing | Yes | Round-robin | None |
| HAMi (CUDA interception) | Yes | Memory + compute quotas | Soft (API-level) |
| gVisor | Yes | No sharing | Moderate |
| Kata Passthrough | Yes | No sharing | Strong |
| **OuterLink (our approach)** | **Yes** | **To be built (R24)** | **To be built** |

---

## Open Questions

1. **MPS + OuterLink integration:** Can we run MPS on each GPU node and route multiple remote users' kernels through MPS for concurrent execution? What's the interaction between LD_PRELOAD interception and MPS?
2. **HAMi coexistence:** Can OuterLink and HAMi coexist on the same Kubernetes cluster, with HAMi handling local sharing and OuterLink handling cross-node pooling?
3. **Isolation boundary:** Where does OuterLink's responsibility for isolation end and the container runtime's begin?

---

## Related Documents

- [R17: Topology-Aware Scheduling](../../phase-09-distributed-os/R17-topology-aware-scheduling/)
- [R23: Heterogeneous GPU Mixing](../../phase-10-ecosystem/R23-heterogeneous-gpu-mixing/)
- [R1: Existing GPU Sharing Projects](../../../../research/R1-existing-projects.md)
- [02-scheduling-and-isolation.md](02-scheduling-and-isolation.md)
- [03-gpu-cloud-architecture.md](03-gpu-cloud-architecture.md)
